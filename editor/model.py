import math
import torch
import torch.nn as nn
import numpy as np
from utils.diffusion_util import diff_CSDI
#from utils.diffusion_util_moe import diff_CSDI as diff_CSDI_moe

class ModelMain(nn.Module):
    def __init__(self, config, device, target_dim=24 * 3):
        super().__init__()
        self.device = device
        self.target_dim = target_dim
        self.is_unconditional = bool(config["model"]["is_unconditional"])

        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        if not self.is_unconditional:
            self.emb_total_dim += 1  # for cond_mask

        self.embed_layer = nn.Embedding(self.target_dim, self.emb_feature_dim)

        cfg_diff = config["diffusion"].copy()
        cfg_diff["side_dim"] = self.emb_total_dim
        cfg_diff["textemb"] = config["model"]["textemb"]

        in_channels = 1 if self.is_unconditional else 2
        if config["diffusion"].get("moe", False):
            self.diffmodel = diff_CSDI_moe(cfg_diff, in_channels, text_mode=config["model"].get("text_mode", "token"))
        else:
            self.diffmodel = diff_CSDI(cfg_diff, in_channels, text_mode=config["model"].get("text_mode", "token"))

        # ---- beta schedule ----
        self.num_steps = cfg_diff["num_steps"]
        self.beta = self._make_beta_schedule(cfg_diff)
        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha, dtype=torch.float32, device=self.device).view(-1, 1, 1)

    def forward(self, batch, is_train=True, text_embedding=None):
        pose_in, pose_gt, tp, mask = self.process_data(batch)
        side_info = self.get_side_info(tp, mask)
        return self._calc_loss(pose_in, pose_gt, mask, side_info, train=is_train, text_emb=text_embedding)

    def evaluate(self, batch, n_samples, text_embedding=None):
        pose_in, pose_gt, tp, mask = self.process_data(batch)
        side_info = self.get_side_info(tp, mask)
        samples = self.impute(pose_in, mask, side_info, n_samples, text_emb=text_embedding)
        return samples, pose_gt, (1 - mask), tp


    def get_side_info(self, tp, cond_mask):
        B, K, L = cond_mask.shape # (B, 24*3, T)
        t_emb = self._time_emb(tp, self.emb_time_dim).unsqueeze(2).expand(-1, -1, K, -1)
        f_emb = self.embed_layer(torch.arange(self.target_dim, device=self.device)).unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        side = torch.cat([t_emb, f_emb], dim=-1).permute(0, 3, 2, 1)  # (B, C, K, L)
        if not self.is_unconditional:
            side = torch.cat([side, cond_mask.unsqueeze(1)], dim=1)
        return side

    def set_input_to_diffmodel(self, x_t, pose_in, cond_mask):
        if self.is_unconditional:
            return x_t.unsqueeze(1)
        obs = pose_in.unsqueeze(1)
        no_tg = ((1 - cond_mask) * x_t).unsqueeze(1)
        return torch.cat([obs, no_tg], dim=1)

    def _calc_loss(self, pose_in, pose_gt, cond_mask, side, train=True, text_emb=None, set_t=-1):
        B, K, L = pose_gt.shape
        t = torch.randint(0, self.num_steps, (B,), device=self.device) if train else torch.full((B,), set_t, dtype=torch.long, device=self.device)
        cur_alpha = self.alpha_torch[t]

        # Add noise to pose_gt
        noise = torch.randn_like(pose_gt)
        x_t = cur_alpha.sqrt() * pose_gt + (1 - cur_alpha).sqrt() * noise

        # pose_in as observed condition
        inp = self.set_input_to_diffmodel(x_t, pose_in, cond_mask)
        pred_eps = self.diffmodel(inp, side, t, text_emb=text_emb)

        loss = ((noise - pred_eps) * (1 - cond_mask)).pow(2).sum()
        denom = (1 - cond_mask).sum().clamp(min=1)
        return loss / denom

    def impute(self, x0, cond_mask, side, n, text_emb=None):
        B, K, L = x0.shape
        alpha_hat = torch.tensor(self.alpha_hat, device=self.device).float()
        alpha = torch.tensor(self.alpha, device=self.device).float()

        outs = torch.zeros(B, n, K, L, device=self.device)
        for s in range(n):
            x_t = torch.randn_like(x0)
            for t in reversed(range(self.num_steps)):
                inp = self.set_input_to_diffmodel(x_t, x0, cond_mask)
                t_vec = torch.full((B,), t, dtype=torch.long, device=self.device)
                eps = self.diffmodel(inp, side, t_vec, text_emb=text_emb)

                coeff1 = 1 / alpha_hat[t].sqrt()
                coeff2 = (1 - alpha_hat[t]) / (1 - alpha[t]).sqrt()
                x_t = coeff1 * (x_t - coeff2 * eps)

                if t > 0:
                    sigma = math.sqrt((1 - alpha[t - 1]) / (1 - alpha[t]) * self.beta[t])
                    x_t += sigma * torch.randn_like(x_t)

            outs[:, s] = x_t * (1 - cond_mask) + x0 * cond_mask
        return outs

    # ====================== Other helpers ==========================

    def process_data(self, batch):
        pose_in = batch["pose_edit"].to(self.device).float().permute(0, 2, 1)
        pose_gt = batch["pose"].to(self.device).float().permute(0, 2, 1)
        tp = batch["timepoints"].to(self.device).float()
        mask = batch["mask"].to(self.device).float().permute(0, 2, 1)
        return pose_in, pose_gt, tp, mask

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
