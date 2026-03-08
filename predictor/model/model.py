import math
import torch
import torch.nn as nn
import numpy as np
from model.diffusion_util import diff_CSDI

class ModelMain(nn.Module):
    def __init__(self, config, device, target_dim=24 * 3):
        super().__init__()
        self.device = device
        self.target_dim = target_dim
        self.is_unconditional = bool(config["model"]["is_unconditional"])
        self.time_step = 0
        
        # MoE load balancing weight
        # self.load_balancing_weight = config["model"].get("load_balancing_weight", 0.01)

        # ---- Embedding: 時間位置編碼 + joint ID 編碼 ----
        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        if not self.is_unconditional:
            self.emb_total_dim += 1  # for cond_mask

        self.embed_layer = nn.Embedding(self.target_dim, self.emb_feature_dim)

        # ---- GRPO / RL config ----
        rl_config = config.get("rl") or {}
        self.sampling_std = float(rl_config.get("sampling_std", 0.05))

        self.balancing_loss = config["model"].get("balance_loss", False)
        # qwen 內部已有 alpha=0.001，預設 weight=1.0 即可；
        # fairscale/tutel 的 l_aux 未縮放，建議設 0.01（對齊 Switch Transformer）
        self.balance_loss_weight = config["model"].get("balance_loss_weight", 0.01)
        print(f"Using load balancing loss: {self.balancing_loss}, weight: {self.balance_loss_weight}")
        # ---- R3 flag: auto-detect or explicit ----
        self.use_r3 = config["model"].get("use_r3", False)
        if self.use_r3:
            print("[R3] Strategy C enabled — two-pass detached routing replay")

        # ---- Diffusion 主體 ----
        cfg_diff = config["diffusion"].copy()
        cfg_diff["side_dim"] = self.emb_total_dim
        cfg_diff["textemb"] = config["model"]["textemb"]

        in_channels = 1 if self.is_unconditional else 2
        if config["model"].get("multirouter", False):
            raise NotImplementedError("Multirouter is not implemented in this codebase. Please set 'multirouter' to False in the config.")
        else:
            print("Using diff_CSDI for MoE without multirouter")
            self.diffmodel = diff_CSDI(cfg_diff, in_channels, text_mode=config["model"].get("text_mode", "token"))
        
        # ---- beta schedule ----
        self.num_steps = cfg_diff["num_steps"]
        self.beta = self._make_beta_schedule(cfg_diff)
        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha, dtype=torch.float32, device=self.device).view(-1, 1, 1)

    def forward(self, batch, is_train=True, text_embedding=None):
        pose, tp, mask = self.process_data(batch)
        side_info = self.get_side_info(tp, mask)
        if self.use_r3 and is_train:
            return self._calc_loss_r3(pose, mask, side_info, text_emb=text_embedding)
        return self._calc_loss(pose, mask, side_info, train=is_train, text_emb=text_embedding)

    def evaluate(self, batch, n_samples, text_embedding=None, noisy_data=None, sample=True):
        pose, tp, mask = self.process_data(batch)
        side_info = self.get_side_info(tp, mask)
        samples = self.impute(pose, mask, side_info, n_samples, text_emb=text_embedding,
                              noisy_data=noisy_data, sample=sample)
        return samples, pose, (1 - mask), tp

    def get_distribution(self, text_emb, batch, t, noisy_data=None):
        """計算並回傳 Normal 分佈 (用於 GRPO single-step)。

        Args:
            noisy_data: 若提供則直接使用，否則自動從 t 生成雜訊。
        """
        observed_data, observed_tp, gt_mask = self.process_data(batch)
        side_info = self.get_side_info(observed_tp, gt_mask)
        # shape: (B, K, L) — K=joints*3, L=frames
        B = observed_data.shape[0]
        frames = observed_data.shape[2]   # L
        joints = observed_data.shape[1] // 3  # K/3

        t = t.to(self.alpha_torch.device)

        if noisy_data is None:
            current_alpha = self.alpha_torch[t].to(self.device)
            noise = torch.randn_like(observed_data)
            noisy_data = current_alpha.sqrt() * observed_data + (1.0 - current_alpha).sqrt() * noise

        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, gt_mask)
        pred = self.diffmodel(total_input, side_info, t, text_emb=text_emb)  # (B, K, L)

        # reshape to (B, frames, joints, 3) for GRPO reward computation
        pred = pred.permute(0, 2, 1)              # (B, L, K)
        pred = pred.reshape(B, frames, joints, 3) # (B, frames, joints, 3)

        mean = pred
        if torch.isnan(mean).any() or torch.isinf(mean).any():
            mean = torch.nan_to_num(mean, nan=0.0, posinf=1.0, neginf=-1.0)

        std = torch.ones_like(mean) * self.sampling_std
        return torch.distributions.Normal(mean, std)

    def get_n_log_prob(self, dist, samples):
        """計算 G 個 samples 在 dist 下的 log-prob，用於 GRPO policy loss。

        Args:
            dist   : Normal(mean (B, frames, joints, 3), std)
            samples: (B, G, frames, joints, 3)

        Returns:
            (B, G) 平均 log prob
        """
        mean = dist.loc.unsqueeze(1)   # (B, 1, frames, joints, 3)
        std  = dist.scale.unsqueeze(1)
        dist_exp = torch.distributions.Normal(mean, std)
        log_prob = dist_exp.log_prob(samples.detach())  # (B, G, frames, joints, 3)
        num_elements = samples.shape[-1] * samples.shape[-2] * samples.shape[-3]
        return log_prob.sum(dim=(-3, -2, -1)) / num_elements  # (B, G)

    def sample_n(self, text_emb, batch, t, G):
        """Single-step: 生成 G 個 samples 及對應 distribution。

        Returns:
            samples: (B, G, frames, joints, 3)
            dist   : Normal
        """
        dist = self.get_distribution(text_emb, batch, t)
        samples = dist.rsample((G,)).permute(1, 0, 2, 3, 4)  # (B, G, frames, joints, 3)
        return samples, dist

    # ====================== 核心模組 ==========================

    # ---- R3 (Strategy C) helpers ----

    def _iter_moe_blocks(self):
        """Yield (key, moe_block) for every FairscaleMoEBlock(_RL) in diffmodel.

        Key format: ``"time_R{r}_L{i}"`` / ``"feature_R{r}_L{i}"``.
        """
        for r, res_block in enumerate(self.diffmodel.residual_layers):
            for prefix, encoder in [("time", getattr(res_block, "time_layer", None)),
                                    ("feature", getattr(res_block, "feature_layer", None))]:
                if encoder is None or not hasattr(encoder, "layers"):
                    continue
                for i, enc_layer in enumerate(encoder.layers):
                    moe_block = getattr(enc_layer, "moe_layer", None)
                    if moe_block is not None:
                        yield f"{prefix}_R{r}_L{i}", moe_block

    def _collect_moe_routing(self):
        """Read ``_last_selected_experts`` from every MoE block (after a forward pass).

        Returns:
            dict[str, Tensor]: key → (N, 2) int64, **detached & on CPU** to avoid
            keeping stale CUDA graph references.
        """
        routing = {}
        for key, block in self._iter_moe_blocks():
            sel = getattr(block, "_last_selected_experts", None)
            if sel is not None:
                routing[key] = sel.detach()  # stay on same device for injection
        return routing

    def _inject_moe_routing(self, routing):
        """Set ``_pending_routing`` on every FairscaleMoEBlock_RL for the next forward.

        Args:
            routing: dict returned by ``_collect_moe_routing()``.
        """
        for key, block in self._iter_moe_blocks():
            if key in routing and hasattr(block, "set_pending_routing"):
                block.set_pending_routing(routing[key])

    def _calc_loss_r3(self, x0, cond_mask, side, text_emb=None):
        """Strategy C — Detached Single‑Step R3 loss.

        Two forward passes on the **same** (x_t, t):
          Pass 1 (no_grad): run diffmodel to record routing decisions.
          Pass 2 (with grad): replay routing via FairscaleMoEBlock_RL masked softmax.

        Compute overhead ≈ +1× forward (no backward on Pass 1).
        """
        B, K, L = x0.shape
        t = torch.randint(0, self.num_steps, (B,), device=self.device)
        cur_alpha = self.alpha_torch[t.to(self.alpha_torch.device)]

        noise = torch.randn_like(x0)
        x_t = cur_alpha.sqrt() * x0 + (1 - cur_alpha).sqrt() * noise
        inp = self.set_input_to_diffmodel(x_t, x0, cond_mask)

        # --- Pass 1: collect I_infer (detached) ---
        with torch.no_grad():
            self.diffmodel(inp, side, t, text_emb=text_emb)
        routing = self._collect_moe_routing()

        # --- Pass 2: R3 replay (with grad) ---
        self._inject_moe_routing(routing)
        pred_eps = self.diffmodel(inp, side, t, text_emb=text_emb)

        # --- Loss ---
        reconstruction_loss = ((noise - pred_eps) * (1 - cond_mask)).pow(2).sum()
        denom = (1 - cond_mask).sum().clamp(min=1)
        reconstruction_loss = reconstruction_loss / denom
        total_loss = reconstruction_loss

        # MoE load balancing loss
        if self.balancing_loss:
            if hasattr(self.diffmodel, "get_load_balancing_loss"):
                load_balancing_loss, _total_fi = self.diffmodel.get_load_balancing_loss()
                total_loss += self.balance_loss_weight * load_balancing_loss

        self.time_step += 1
        return total_loss

    def get_side_info(self, tp, cond_mask):
        B, K, L = cond_mask.shape
        t_emb = self._time_emb(tp, self.emb_time_dim).unsqueeze(2).expand(-1, -1, K, -1)
        f_emb = self.embed_layer(torch.arange(self.target_dim, device=self.device)).unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        side = torch.cat([t_emb, f_emb], dim=-1).permute(0, 3, 2, 1)  # (B, C, K, L)
        if not self.is_unconditional:
            side = torch.cat([side, cond_mask.unsqueeze(1)], dim=1)
        return side

    def set_input_to_diffmodel(self, x_t, x0, cond_mask):
        if self.is_unconditional:
            return x_t.unsqueeze(1)
        obs = (cond_mask * x0).unsqueeze(1)
        no_tg = ((1 - cond_mask) * x_t).unsqueeze(1)
        return torch.cat([obs, no_tg], dim=1)

    def _calc_loss(self, x0, cond_mask, side, train=True, text_emb=None, set_t=-1):
        B, K, L = x0.shape
        t = torch.randint(0, self.num_steps, (B,), device=self.device) if train else torch.full((B,), set_t, dtype=torch.long, device=self.device)
        cur_alpha = self.alpha_torch[t.to(self.alpha_torch.device)]

        noise = torch.randn_like(x0)
        x_t = cur_alpha.sqrt() * x0 + (1 - cur_alpha).sqrt() * noise

        inp = self.set_input_to_diffmodel(x_t, x0, cond_mask)
        pred_eps = self.diffmodel(inp, side, t, text_emb=text_emb)
        # exit(0)

        # 主要的重建損失
        reconstruction_loss = ((noise - pred_eps) * (1 - cond_mask)).pow(2).sum()
        denom = (1 - cond_mask).sum().clamp(min=1)
        reconstruction_loss = reconstruction_loss / denom
        total_loss = reconstruction_loss
        # MoE load balancing loss (如果存在)
        if self.balancing_loss:
            if hasattr(self.diffmodel, 'get_load_balancing_loss'):
                load_balancing_loss, _total_fi = self.diffmodel.get_load_balancing_loss()
                total_loss += self.balance_loss_weight * load_balancing_loss
                
        self.time_step += 1
        return total_loss

    def impute(self, x0, cond_mask, side, n, text_emb=None, noisy_data=None, sample=True):
        """Full denoising chain.

        Args:
            noisy_data: (B, K, L) or (B, n, K, L) — 若給定則以此為初始雜訊。
            sample    : 是否在每步加入隨機雜訊（True = 標準 DDPM）。
        """
        B, K, L = x0.shape
        alpha_hat = torch.tensor(self.alpha_hat, device=self.device).float()
        alpha = torch.tensor(self.alpha, device=self.device).float()

        outs = torch.zeros(B, n, K, L, device=self.device)
        for s in range(n):
            if noisy_data is not None:
                if noisy_data.ndim == 4 and noisy_data.shape[1] == n:
                    x_t = noisy_data[:, s].clone()
                elif noisy_data.ndim == 3:
                    x_t = noisy_data.clone()
                else:
                    x_t = torch.randn_like(x0)
            else:
                x_t = torch.randn_like(x0)

            for t in reversed(range(self.num_steps)):
                inp = self.set_input_to_diffmodel(x_t, x0, cond_mask)
                t_vec = torch.full((B,), t, dtype=torch.long, device=self.device)
                eps = self.diffmodel(inp, side, t_vec, text_emb=text_emb)

                coeff1 = 1 / alpha_hat[t].sqrt()
                coeff2 = (1 - alpha_hat[t]) / (1 - alpha[t]).sqrt()
                x_t = coeff1 * (x_t - coeff2 * eps)

                if t > 0 and sample:
                    sigma = math.sqrt((1 - alpha[t - 1]) / (1 - alpha[t]) * self.beta[t])
                    x_t += sigma * torch.randn_like(x_t)

            outs[:, s] = x_t * (1 - cond_mask) + x0 * cond_mask
        return outs

    # ====================== GRPO / Trajectory ==========================

    def sample_trajectory(self, text_emb, batch, G):
        """全去噪鏈取樣，回傳 G 條軌跡供 GRPO 使用。

        當 ``use_r3=True`` 時，額外收集每步的 MoE routing 決策，
        供 ``backprop_trajectory_loss`` 在 Pass 2 做 R3 replay。

        Returns:
            final_samples  : (B, G, K, L)
            total_log_probs: (B, G)
            all_latents    : list[Tensor (B, G, K, L)], length = num_steps+1
                             [x_T, x_{T-1}, ..., x_0]
            all_routing    : list[dict] | None
                             若 ``use_r3``，長度 = num_steps，迭代順序同 loop
                             （即 index 0 = t=T-1, index -1 = t=0）。
                             每個 dict: key → (N, 2) int64 detached tensor。
                             若非 R3 模式，回傳 ``None``。
        """
        observed_data, observed_tp, gt_mask = self.process_data(batch)
        side_info = self.get_side_info(observed_tp, gt_mask)
        B, K, L = observed_data.shape

        x_t = torch.randn(B, 1, K, L, device=self.device).expand(-1, G, -1, -1).clone()
        all_latents = [x_t]
        total_log_probs = torch.zeros(B, G, device=self.device)
        all_routing = [] if self.use_r3 else None

        alpha_hat = torch.tensor(self.alpha_hat, device=self.device).float()
        alpha = torch.tensor(self.alpha, device=self.device).float()

        side_rep  = side_info.repeat_interleave(G, dim=0)
        cond_rep  = gt_mask.repeat_interleave(G, dim=0)
        obs_rep   = observed_data.repeat_interleave(G, dim=0)
        if isinstance(text_emb, tuple):
            text_rep = tuple(u.repeat_interleave(G, dim=0) for u in text_emb)
        else:
            text_rep = text_emb.repeat_interleave(G, dim=0) if text_emb is not None else None

        for t in reversed(range(self.num_steps)):
            x_t_flat = x_t.reshape(B * G, K, L)
            inp = self.set_input_to_diffmodel(x_t_flat, obs_rep, cond_rep)
            t_vec = torch.full((B * G,), t, dtype=torch.long, device=self.device)

            eps  = self.diffmodel(inp, side_rep, t_vec, text_emb=text_rep)

            # R3: 收集本步 routing（推理 Pass 1）
            if self.use_r3:
                all_routing.append(self._collect_moe_routing())

            c1   = 1 / alpha_hat[t].sqrt()
            c2   = (1 - alpha_hat[t]) / (1 - alpha[t]).sqrt()
            mean = c1 * (x_t_flat - c2 * eps)

            if t > 0:
                sigma = math.sqrt((1 - alpha[t - 1]) / (1 - alpha[t]) * self.beta[t])
                z     = torch.randn_like(mean)
                x_prev = mean + sigma * z
                step_lp = (-0.5 * z ** 2 - math.log(sigma)).sum(dim=(1, 2))
                total_log_probs += step_lp.reshape(B, G)
                x_t = x_prev.reshape(B, G, K, L)
            else:
                x_t = mean.reshape(B, G, K, L)

            all_latents.append(x_t)
            # 關鍵：截斷梯度，避免深度為 T 的計算圖
            x_t = x_t.detach()

        final_samples = x_t * (1 - gt_mask.unsqueeze(1)) + observed_data.unsqueeze(1) * gt_mask.unsqueeze(1)
        return final_samples, total_log_probs, all_latents, all_routing

    def get_trajectory_log_prob(self, all_latents, text_emb, batch):
        """計算給定軌跡在本模型（作為 reference）下的 log-prob。

        Args:
            all_latents: [x_T, ..., x_0], len = num_steps+1, each (B, G, K, L)

        Returns:
            (B, G) total log prob
        """
        observed_data, observed_tp, gt_mask = self.process_data(batch)
        side_info = self.get_side_info(observed_tp, gt_mask)
        B, K, L = observed_data.shape
        G = all_latents[0].shape[1]

        alpha_hat = torch.tensor(self.alpha_hat, device=self.device).float()
        alpha     = torch.tensor(self.alpha,     device=self.device).float()

        side_rep = side_info.repeat_interleave(G, dim=0)
        cond_rep = gt_mask.repeat_interleave(G, dim=0)
        obs_rep  = observed_data.repeat_interleave(G, dim=0)
        if isinstance(text_emb, tuple):
            text_rep = tuple(u.repeat_interleave(G, dim=0) for u in text_emb)
        else:
            text_rep = text_emb.repeat_interleave(G, dim=0) if text_emb is not None else None

        total_lp = torch.zeros(B, G, device=self.device)

        for t in reversed(range(self.num_steps)):
            idx_curr = self.num_steps - (t + 1)
            idx_next = self.num_steps - t
            x_t    = all_latents[idx_curr]
            x_prev = all_latents[idx_next]

            x_t_flat = x_t.reshape(B * G, K, L)
            inp  = self.set_input_to_diffmodel(x_t_flat, obs_rep, cond_rep)
            t_vec = torch.full((B * G,), t, dtype=torch.long, device=self.device)

            eps  = self.diffmodel(inp, side_rep, t_vec, text_emb=text_rep)
            c1   = 1 / alpha_hat[t].sqrt()
            c2   = (1 - alpha_hat[t]) / (1 - alpha[t]).sqrt()
            mean = c1 * (x_t_flat - c2 * eps)

            sigma = math.sqrt((1 - alpha[t - 1]) / (1 - alpha[t]) * self.beta[t])
            x_prev_flat = x_prev.reshape(B * G, K, L)
            mse  = (x_prev_flat - mean) ** 2
            lp   = (-0.5 * mse / sigma ** 2 - math.log(sigma)).sum(dim=(1, 2))
            total_lp += lp.reshape(B, G)

        return total_lp

    def backprop_trajectory_loss(self, all_latents, text_emb, batch, advantages,
                                  all_routing=None):
        """Memory-efficient GRPO backprop：逐步計算 log-prob 並立即 backward。

        當 ``all_routing`` 不為 None 時，在每一步的 diffmodel forward 前
        注入推理階段記錄的 routing，讓 FairscaleMoEBlock_RL 走 R3 masked
        softmax，確保訓練梯度使用與推理相同的專家選擇。

        Args:
            all_latents : list from ``sample_trajectory``
            advantages  : (B, G) — 歸一化後的 advantage
            all_routing : list[dict] | None — 來自 ``sample_trajectory``
                          的 R3 routing，長度 = num_steps，與 loop 迭代順
                          序一致（index 0 = t=T-1）。
        """
        observed_data, observed_tp, gt_mask = self.process_data(batch)
        side_info = self.get_side_info(observed_tp, gt_mask)
        B, K, L = observed_data.shape
        G = all_latents[0].shape[1]

        alpha_hat = torch.tensor(self.alpha_hat, device=self.device).float()
        alpha     = torch.tensor(self.alpha,     device=self.device).float()

        side_rep = side_info.repeat_interleave(G, dim=0).detach()
        cond_rep = gt_mask.repeat_interleave(G, dim=0).detach()
        obs_rep  = observed_data.repeat_interleave(G, dim=0).detach()
        if isinstance(text_emb, tuple):
            text_rep = tuple(u.repeat_interleave(G, dim=0).detach() for u in text_emb)
        else:
            text_rep = text_emb.repeat_interleave(G, dim=0).detach() if text_emb is not None else None

        adv_flat = advantages.view(-1)  # (B*G,)

        step_idx = 0
        for t in reversed(range(self.num_steps)):
            idx_curr = self.num_steps - (t + 1)
            idx_next = self.num_steps - t
            x_t    = all_latents[idx_curr]
            x_prev = all_latents[idx_next]

            x_t_flat = x_t.reshape(B * G, K, L).detach()
            inp  = self.set_input_to_diffmodel(x_t_flat, obs_rep, cond_rep)
            t_vec = torch.full((B * G,), t, dtype=torch.long, device=self.device)

            # R3: 注入推理時的 routing（Pass 2 replay）
            if all_routing is not None and step_idx < len(all_routing):
                self._inject_moe_routing(all_routing[step_idx])

            eps  = self.diffmodel(inp, side_rep, t_vec, text_emb=text_rep)
            c1   = 1 / alpha_hat[t].sqrt()
            c2   = (1 - alpha_hat[t]) / (1 - alpha[t]).sqrt()
            mean = c1 * (x_t_flat - c2 * eps)

            sigma = math.sqrt((1 - alpha[t - 1]) / (1 - alpha[t]) * self.beta[t])
            x_prev_flat = x_prev.reshape(B * G, K, L)
            mse  = (x_prev_flat - mean) ** 2
            lp   = (-0.5 * mse / sigma ** 2 - math.log(sigma)).sum(dim=(1, 2))

            loss_step = -(lp * adv_flat).mean()
            loss_step.backward()

            del loss_step, lp, eps, mean, inp
            step_idx += 1

    # ====================== 其他輔助 ==========================

    def process_data(self, batch):
        pose = batch["pose"].to(self.device).float().permute(0, 2, 1)
        #print("pose shape:", pose.shape)
        #exit(0)
        tp = batch["timepoints"].to(self.device).float()
        mask = batch["mask"].to(self.device).float().permute(0, 2, 1)
        return pose, tp, mask

    def _time_emb(self, pos, d_model):
        pe = torch.zeros_like(pos).unsqueeze(-1).repeat(1, 1, d_model)
        div = torch.exp(torch.arange(0, d_model, 2, device=self.device) * (-math.log(10000.0) / d_model))
        pe[..., 0::2] = torch.sin(pos.unsqueeze(-1) * div)
        pe[..., 1::2] = torch.cos(pos.unsqueeze(-1) * div)
        return pe

    def _make_beta_schedule(self, cfg_diff):
        schedule = cfg_diff["schedule"]
        if schedule == "quad":
            return np.linspace(cfg_diff["beta_start"] ** 0.5, cfg_diff["beta_end"] ** 0.5, self.num_steps) ** 2
        elif schedule == "linear":
            return np.linspace(cfg_diff["beta_start"], cfg_diff["beta_end"], self.num_steps)
        elif schedule == "cosine":
            return self._betas_for_alpha_bar(
                self.num_steps,
                lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
            )
        else:
            raise ValueError(f"Unknown beta schedule: {schedule}")

    def _betas_for_alpha_bar(self, T, alpha_bar, max_beta=0.5):
        return np.array([min(1 - alpha_bar((i + 1) / T) / alpha_bar(i / T), max_beta) for i in range(T)])
