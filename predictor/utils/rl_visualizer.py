import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import imageio
import os
from typing import Dict, List, Tuple

# Use Agg backend to avoid display issues
matplotlib.use('Agg')

# SMPL_24 Edges (Same as FineFS)
EDGES = [
    (0, 1), (1, 4), (4, 7), (7, 10), (0, 2), (2, 5), (5, 8), (8, 11),
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15), (12, 13), (13, 16), (16, 18),
    (18, 20), (20, 22), (12, 14), (14, 17), (17, 19), (19, 21), (21, 23)
]

class RLVisualizer:
    def __init__(self, output_dir: str, val_idx: int = 0, 
                 top_k: int = 3, bot_k: int = 3, viz_mode: str = "all", device: str = "cpu", joints: int = 24):
        self.output_dir = os.path.join(output_dir, "visualize")
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.val_idx = val_idx
        self.top_k = top_k
        self.bot_k = bot_k
        self.viz_mode = viz_mode
        self.device = device
        self.joints = joints
        
        if self.joints == 24:
            self.edges = EDGES
        elif self.joints == 17:
            self.edges = [
                (0, 1), (1, 2), (2, 3),        # r-leg
                (0, 4), (4, 5), (5, 6),        # l-leg
                (0, 7), (7, 8), (8, 9), (9,10),# spine/head
                (8,11), (11,12), (12,13),      # l-arm
                (8,14), (14,15), (15,16)       # r-arm
            ]
        else:
            self.edges = []
        
        # Cache for the fixed validation sample
        self.fixed_sample: Dict = None

    def load_fixed_sample(self, val_dataset, target_path=None):
        """
        Loads and caches the specific i-th sample from validation set.
        If target_path is provided, it searches for the sample with that file path.
        """
        found_idx = -1
        
        # 1. Try to find by path if provided
        if target_path:
            print(f"[RLVisualizer] Searching for sample with path: {target_path}")
            # Assuming val_dataset has a way to access file paths or we iterate
            # FineFS dataset usually has `data` list of dicts or similar. 
            # Let's try to access the internal list if possible, or iterate.
            # Based on FineFS implementation, dataset[i] returns a dict, but maybe doesn't give full path easily?
            # Usually dataset.data is a list of metadata.
            try:
                # Iterate to find match
                # Access FineFS-specific attributes: data_idx stores (key, start_time)
                # file_paths stores (key -> path)
                if hasattr(val_dataset, 'data_idx') and hasattr(val_dataset, 'file_paths'):
                    for i, (key, _) in enumerate(val_dataset.data_idx):
                        item_path = val_dataset.file_paths.get(key, "")
                        
                        # Match end of path/substring to be robust
                        # Note: YAML provided .mp4 but dataset has .pk. 
                        # We should match the parent folder or filename without extension if possible.
                        # Simple substring match is safest.
                        if target_path in item_path:
                            found_idx = i
                            break
                else:
                    print("[RLVisualizer] Dataset is missing file_paths attribute. Skipping path search.")
                    
            except Exception as e:
                print(f"[RLVisualizer] Search failed: {e}")

        if found_idx != -1:
            print(f"[RLVisualizer] Found target sample at index {found_idx}!")
            self.val_idx = found_idx
        elif target_path:
            print(f"[RLVisualizer] ⚠️ Target path not found. Falling back to val_idx {self.val_idx}.")

        # 2. Index Boundary Check
        if self.val_idx >= len(val_dataset):
            print(f"[RLVisualizer] Warning: val_idx {self.val_idx} out of range (len={len(val_dataset)}). Using len-1.")
            self.val_idx = len(val_dataset) - 1
            
        sample = val_dataset[self.val_idx]
        
        # Convert to batch format (add batch dim) WITHOUT permute
        # Standard format from dataset is (T, K)
        # We want (1, T, K) for model input
        pose = torch.from_numpy(sample["pose"]).float().unsqueeze(0).to(self.device) # (1, T, K)
        tp = torch.from_numpy(sample["timepoints"]).float().unsqueeze(0).to(self.device)
        mask = torch.from_numpy(sample["mask"]).float().unsqueeze(0).to(self.device) # (1, T, K)
        
        # Mask for generation: Use the official mask (1 for observed input, 0 for target)
        # This fixes the "jumping" issue at frame 30, as it enforces GT for t < 30.
        gen_mask = mask.clone()
        
        self.fixed_sample = {
            "pose": pose,      # GT Pose (1, T, K)
            "tp": tp,
            "mask": mask,       # Original Mask (1, T, K)
            "gen_mask": gen_mask, # Validation Mask (1 for input, 0 for output)
            "motion_name": sample["motion_name"],
            "raw_pose_T_K": sample["pose"] # Keep raw numpy (T, K) for plotting
        }
        
        print(f"[RLVisualizer] Loaded fixed sample '{sample['motion_name']}' (Index {self.val_idx})")

    @torch.no_grad()
    def run_epoch_viz(self, epoch: int, model, reward_model, text_encoder, current_std: float, num_variants: int):
        """
        Main function to run visualization for the epoch.
        Supports multiple modes controlled by `viz_mode`:
        
        1. 'inference_exploration' (Standard):
           - Full Diffusion Generation (50 steps).
           - Variants differ by *both* Initial Noise and Step Noise.
           - Represents standard user experience and model diversity.
           
        2. 'training_exploration' (RL View):
           - Single Training Step (e.g. t=20).
           - Variants differ by `sampling_std` around the mean prediction.
           - Represents what the RL optimizer sees and updates.
           
        3. 'inference_stability' (Stability View):
           - Full Diffusion Generation.
           - Variants share the SAME Initial Noise, but have different Step Noise.
           - Isolates the effect of stochastic denoising path vs initial condition.
           - If trajectories are very different, the generative process is unstable/high-variance.
           
        4. 'all': Runs all of the above.
        """
        if self.fixed_sample is None:
            print("[RLVisualizer] Error: Fixed sample not loaded. Call load_fixed_sample() first.")
            return

        print(f"[RLVisualizer] Generating visualization for Epoch {epoch} (Mode: {self.viz_mode})...")
        
        # [OOM Fix] Offload Reward Model to CPU to free up VRAM for Generation
        # We must restore it to GPU at the end.
        reward_model.cpu()
        torch.cuda.empty_cache()
        
        try:
            # Check Modes
            # [已禁用] Inference Exploration - 每条轨迹起点不同，用于观察多样性
            # run_inference = self.viz_mode in ["inference_exploration", "all"]
            run_training = self.viz_mode in ["training_exploration", "all"]
            run_stability = self.viz_mode in ["inference_stability", "all"]
            
            # 1. Inference Exploration (已禁用 - 保留 Training 和 Stability 即可)
            # if run_inference:
            #     try:
            #         self._run_inference_exploration(epoch, model, reward_model, text_encoder, current_std, num_variants)
            #     except Exception as e:
            #         print(f"❌ Inference Viz Failed: {e}")
            #         import traceback
            #         traceback.print_exc()

            # 2. Training Exploration
            if run_training:
                try:
                    self._run_training_exploration(epoch, model, reward_model, text_encoder, current_std, num_variants)
                except Exception as e:
                    print(f"❌ Training Viz Failed: {e}")
                    import traceback
                    traceback.print_exc()
                    
            # 3. Inference Stability
            if run_stability:
                try:
                    self._run_inference_stability(epoch, model, reward_model, text_encoder, current_std, num_variants)
                except Exception as e:
                    print(f"❌ Stability Viz Failed: {e}")
                    
        finally:
            # [Restore] Always move Reward Model back to GPU for training
            print("[RLVisualizer] Restoring Reward Model to GPU...")
            reward_model.to(self.device)
            torch.cuda.empty_cache()

    def _prepare_common_inputs(self, text_encoder):
        # 準備共用輸入
        pose = self.fixed_sample["pose"]
        tp = self.fixed_sample["tp"]
        gen_mask = self.fixed_sample["gen_mask"]
        motion_name = self.fixed_sample["motion_name"]

        # 文本條件 (Text Embedding)
        tok_emb, tok_mask = text_encoder([motion_name])
        tok_emb, tok_mask = tok_emb.to(self.device), tok_mask.to(self.device)
        text_cond = (tok_emb, tok_mask)
        
        feed = {"pose": pose, "mask": gen_mask, "timepoints": tp}
        return feed, text_cond, pose

    def _rank_and_render(self, epoch, mode_name, variants, reward_model, num_variants, pure_pred=None):
        """
        對生成的變體進行 Reward 排序並準備渲染資料。
        
        Args:
            pure_pred (Tensor, optional): 不含隨機性的純預測 (Mean) 軌跡，將繪製為紅色。 Shape: (T, K)
        """
        # Reward Model 需要 GT 資料: (1, 1, T, J, 3)
        gt_pose = self.fixed_sample["pose"] # (1, T, K)
        gt_5d = gt_pose.reshape(1, 1, gt_pose.shape[1], self.joints, 3)
        
        # 處理 Variants: 確保在 CPU 上以節省顯存
        if variants.device != torch.device("cpu"):
            variants = variants.detach().cpu()
            
        variants_TK = variants.permute(0, 2, 1) # (G, T, K)
        
        # 在 CPU 上分批計算 Reward (避免 OOM)
        rewards_list = []
        gt_batch_cpu = gt_5d.cpu()
        
        with torch.no_grad():
            for i in range(num_variants):
                # 取出一個變體: (1, 1, T, J, 3)
                v_batch = variants_TK[i:i+1].reshape(1, 1, variants_TK.shape[1], self.joints, 3)
                
                # 計算 Reward (Model 與 Data 都在 CPU)
                r, _ = reward_model(v_batch, gt_batch_cpu)
                rewards_list.append(r.item())
                
        rewards = np.array(rewards_list)
        
        # 排序
        sorted_idx = np.argsort(rewards)
        best_idx = sorted_idx[-self.top_k:][::-1] # 取最高分
        worst_idx = sorted_idx[:self.bot_k]       # 取最低分
        
        # 收集軌跡資料
        trajectories = []
        offset_step = 0.5 # 每個軌跡在 X 軸的偏移量，避免重疊
        
        # 1. 真實動作 (GT) - 藍色
        gt_data = self.fixed_sample["raw_pose_T_K"].reshape(-1, self.joints, 3)
        trajectories.append({
            "data": gt_data, "color": "blue", "alpha": 1.0, 
            "label": "GT (Ground Truth)", "linewidth": 2.0, "offset": np.array([0, 0, 0])
        })

        # 2. 純預測 (Pure Prediction) - 紅色
        # 這是模型在不加隨機採樣情況下的「平均」或「最佳」預測
        if pure_pred is not None:
            if pure_pred.device != torch.device("cpu"):
                pure_pred = pure_pred.detach().cpu()
            pure_data = pure_pred.numpy().reshape(-1, self.joints, 3)
            # 偏移量設為稍微偏一點，避免完全蓋住 GT
            off = np.array([offset_step * 0.5, 0, 0])
            trajectories.append({
                "data": pure_data, "color": "red", "alpha": 0.9,
                "label": "Pure Pred (Mean)", "linewidth": 2.0, "offset": off
            })
        
        # 3. 高分變體 (Top K) - 綠色系
        top_colors = ["darkgreen", "indigo", "saddlebrown", "darkcyan", "darkslategray"]
        for i, idx in enumerate(best_idx):
            data = variants[idx].detach().permute(1, 0).cpu().numpy().reshape(-1, self.joints, 3)
            # 偏移量向右遞增
            off = np.array([(i + 1) * offset_step, 0, 0])
            trajectories.append({
                "data": data, "color": top_colors[i%len(top_colors)], "alpha": 0.7,
                "label": f"Top-{i+1} ({rewards[idx]:.2f})", "linewidth": 1.5, "offset": off
            })
            
        # 4. 低分變體 (Bot K) - 亮色/橘色系
        bot_colors = ["lime", "orchid", "orange", "cyan", "gold"]
        for i, idx in enumerate(worst_idx):
            data = variants[idx].detach().permute(1, 0).cpu().numpy().reshape(-1, self.joints, 3)
            # 偏移量向左遞增
            off = np.array([-(i + 1) * offset_step, 0, 0])
            trajectories.append({
                "data": data, "color": bot_colors[i%len(bot_colors)], "alpha": 0.5,
                "label": f"Bot-{i+1} ({rewards[idx]:.2f})", "linewidth": 1.0, "offset": off
            })
            
        self.render_video(epoch, trajectories, gt_data, mode_name=mode_name)

    def _run_inference_exploration(self, epoch, model, reward_model, text_encoder, current_std, num_variants, mode_name="inference"):
        #目前不用
        torch.cuda.empty_cache() # [Fix] 清理顯存
        
        # 固定種子以確保可重現性
        # 處理 epoch 可能是整數或字符串 (如 "1_batch100")
        epoch_num = int(str(epoch).split('_')[0]) if isinstance(epoch, str) else epoch
        torch.manual_seed(2049 + epoch_num) 
        
        feed, text_cond, _ = self._prepare_common_inputs(text_encoder)
        
        # 生成變體 (分批生成以避免 OOM)
        # Inference 模式下，是一個完整的 50 步去噪過程
        # 這裡不傳入 noisy_data，讓 model 內部隨機生成不同的初始噪聲 (Diversity)
        collected_variants = []
        
        for _ in range(num_variants):
            # Evaluate 回傳 (1, G, K, L) -> 這裡 G=1 (Single Batch)
            p_out = model.evaluate(feed, 1, text_embedding=text_cond)[0] # (1, 1, K, L)
            collected_variants.append(p_out[0, 0].cpu()) # (K, L) -> 立刻搬回 CPU
            
        # 堆疊變體 -> (G, K, T)
        variants = torch.stack(collected_variants, dim=0) 
        
        # Inference 模式：不需要紅線 (Pure Prediction)，因為每個變體的初始點都不同，沒有單一基準
        self._rank_and_render(epoch, mode_name, variants, reward_model, num_variants, pure_pred=None)

    def _run_training_exploration(self, epoch, model, reward_model, text_encoder, current_std, num_variants):
        """
        ═══════════════════════════════════════════════════════════════════
        模式二：Training Exploration（训练探索）
        ═══════════════════════════════════════════════════════════════════
        
        【目的】
        展示训练时"实际发生的采样过程"，即模型在 RL 训练循环中看到的轨迹。
        
        【设定】
        - 固定 seed = 42（确保可重现）
        - 共享同一个初始噪声 x_T
        - 从 x_T 开始，通过 Stochastic Sampling 生成 G 条不同的轨迹（分岔）
        - 使用 model.sample_trajectory（与训练逻辑完全一致）
        
        【颜色含义】
        🟢 深绿色系（darkgreen, indigo 等）：Top K 高 Reward 的动作
           - 这些是训练时会被"奖励"的行为
           - 模型会学习增加这类轨迹的生成概率
           - 按分数从高到低排列
           
        � 亮色系（lime, orchid, orange 等）：Bot K 低 Reward 的动作  
           - 这些是训练时会被"惩罚"的行为
           - 模型会学习降低这类轨迹的生成概率
           - 颜色较亮，便于与高分轨迹区分
           
        🔵 蓝色线（Ground Truth）：参考答案
           - 展示理想的动作序列
           
        【如何解读】
        - 随着训练进行，深绿色轨迹（高分）应该：
          1. 越来越接近 GT（蓝线）
          2. 与亮色轨迹（低分）的差距越来越大（分化明显）
        - 如果深绿和亮色混在一起 → 模型还在探索，没有明确的"好坏"概念
        ═══════════════════════════════════════════════════════════════════
        """
        torch.cuda.empty_cache()
        torch.manual_seed(123)
        
        feed, text_cond, gt_pose = self._prepare_common_inputs(text_encoder)
        
        # 1. Generate Variants (G paths from same x_T)
        # final_samples: (1, G, K, L)
        # text_cond is already a tuple (tok_emb, tok_mask), pass it directly
        final_samples, _, _ = model.sample_trajectory(text_cond, feed, G=num_variants)
        variants = final_samples[0] # (G, K, L)
        
        # 2. Render（传入 variants，render 函数会根据 reward 排序并上色）
        self._rank_and_render(epoch, "training_trajectory", variants, reward_model, num_variants, pure_pred=None)

    def _run_inference_stability(self, epoch, model, reward_model, text_encoder, current_std, num_variants):
        """
        ═══════════════════════════════════════════════════════════════════
        模式三：Inference Stability（推理稳定性）⭐ 观察模型成长的主要指标
        ═══════════════════════════════════════════════════════════════════
        
        【目的】
        在"完全相同的条件"下，观察模型在不同训练阶段的表现变化。
        这是评估模型是否真正"学到东西"的最直接方式。
        
        【设定】
        - 完全固定 seed = 42（不加 epoch，所有 epoch 都用相同的随机序列）
        - 固定的初始噪声 x_T
        - 固定的输入样本
        ⚠️ 关键：跨 epoch 的条件完全相同，确保公平比较
        
        【颜色含义】
        🔴 红色线（Pure Prediction）：模型的"最确定答案"
           - 去噪过程中完全不加随机噪声（Deterministic / DDIM 模式）
           - 代表模型在当前训练阶段的"最佳预测"
           - 这是模型的"最自信输出"
           
        🟢 绿色轨迹（Stochastic Variants）：加了随机性的结果
           - 去噪过程中每步都加随机噪声（DDPM 模式）
           - 但因为 seed 固定，所以随机序列每次都相同
           - 显示模型在"标准采样"下的输出
           - 通常会围绕红线波动
           
        🔵 蓝色线（Ground Truth）：参考答案
           - 理想的动作序列
           
        【如何解读】
        ✅ 模型在进步的表现：
           - 红线（Pure）随着 epoch 增加，越来越接近蓝线（GT）
           - 绿线（Variants）的分数（Reward）越来越高
           - 红线和绿线的位置相对稳定（鲁棒性好）
           
        ❌ 模型未收敛的表现：
           - 红线在不同 epoch 之间跳来跳去（不稳定）
           - 绿线和红线距离很远（模型对随机性很敏感）
           - 分数没有明显提升
           
        【跨 epoch 比较】
        由于所有条件完全固定，你可以直接对比：
        - Epoch 1 的红线 vs Epoch 10 的红线
        - 如果模型在学习，Epoch 10 的红线应该明显更好
        ═══════════════════════════════════════════════════════════════════
        """
        torch.cuda.empty_cache()
        # [重要] 完全固定 seed (不加 epoch)，确保所有 epoch 使用相同的初始噪声和随机序列
        # 这样才能公平比较模型在不同训练阶段的表现
        torch.manual_seed(123)
        
        feed, text_cond, _ = self._prepare_common_inputs(text_encoder)
        
        # 1. 生成固定的初始噪聲
        # feed["pose"] shape: (1, T, K)
        B, T, K = feed["pose"].shape
        # model.impute 預期 noisy_data 形狀為 (B, K, L)
        fixed_noise = torch.randn((B, K, T), device=self.device)
        
        # 2. 計算 Pure Prediction (紅線) - Deterministic (sample=False)
        # 呼叫 evaluate, 傳入 noisy_data, sample=False
        p_pure = model.evaluate(feed, 1, text_embedding=text_cond, noisy_data=fixed_noise, sample=False)[0]
        pure_pred = p_pure[0, 0].cpu() # (K, L)
        pure_pred = pure_pred.permute(1, 0) # (T, K) 轉為時間軸在前的格式
        
        # 3. 生成 Variants (綠線) - Stochastic (sample=True)
        collected_variants = []
        for _ in range(num_variants):
            # 每次都傳入相同的 fixed_noise，但 sample=True
            p_out = model.evaluate(feed, 1, text_embedding=text_cond, noisy_data=fixed_noise, sample=True)[0]
            collected_variants.append(p_out[0, 0].cpu())
        
        variants = torch.stack(collected_variants, dim=0) # (G, K, T)
        
        self._rank_and_render(epoch, "inference_stability", variants, reward_model, num_variants, pure_pred=pure_pred)

    def render_video(self, epoch, trajectories, gt_pose, mode_name="default"):
        """
        Renders the accumulated trajectories into a video.
        """
        frames = []
        seq_len = gt_pose.shape[0] # Should be 90 (30 input + 40 output + padding?) or 70
        input_n = 30
        
        # [Fix] Ensure generated variants start from GT input (t=0 to 30) for visual continuity
        # The diffusion model might output slightly different values even for conditioned frames if not enforced.
        # But `model.impute` uses `cond_mask` to enforce observed data. 
        # If it "jumps" at t=30, it might be that `gen_mask` was zeros?
        # In `load_fixed_sample`, gen_mask = torch.zeros_like(mask).
        # This means generation is UNCONDITIONED (In-painting mask is all 0 means generate everything?).
        # Wait, `impute` logic uses `cond_mask` to overwrite `total_input`.
        # If `gen_mask` is all zeros, then NOTHING is preserved?
        # Ah! `model.impute` code:
        # `model_input = (1 - cond_mask) * noisy_data + cond_mask * observed_data`
        # If `cond_mask` is 0, it uses `noisy_data` (generated).
        # So we definitely want to visualize the model generating FROM scratch? 
        # USER said: "後40frame炸開". Before that should be GT?
        # If we pass `gen_mask` as all zeros, the model regenerates 0-30 too?
        # Let's check `load_fixed_sample`:
        # `gen_mask = torch.zeros_like(mask)`
        # `mask` from dataset is typically 1 for observed (0-30) and 0 for unobserved (30-70).
        # IF we want conditioned generation, we should use the original `mask`!
        # `feed` uses `gen_mask`.
        
        # Setup Figure once
        fig = plt.figure(figsize=(12, 8)) # Wider for legend
        ax = fig.add_subplot(111, projection='3d') 
        # But user also mentioned "畫出座標軸和scale". 
        # Let's stick to 2D for clarity if user wants "XY plane front view", 
        # BUT 3D is better for "slightly tilted". 
        # User said: "正對面(XY平面) + 稍微傾斜的3D視角" -> 3D Plot with fixed ViewInit.
        
        # Determine strict bounds to prevent camera jumping
        # Use GT to define "normal" bounds, maybe expand slightly
        # [Fix] Include offset in bounds calculation
        all_coords_list = []
        for t in trajectories:
            # Apply offset to data for bounds calculation
            # t["data"] is (T, 24, 3)
            # t["offset"] is (3,)
            adjusted_data = t["data"] + t["offset"]
            all_coords_list.append(adjusted_data)
        
        all_coords = np.concatenate(all_coords_list, axis=0) # (Total_Frames, J, 3)
        min_vals = np.min(all_coords, axis=(0, 1))
        max_vals = np.max(all_coords, axis=(0, 1))
        
        # Force bounds to be at least [-1, 1] for stability if motions are small
        bound = 1.0
        x_lim = [min(-bound, min_vals[0]), max(bound, max_vals[0])]
        y_lim = [min(-bound, -max_vals[1]), max(bound, -min_vals[1])] # Flipped Y
        z_lim = [min(-bound, min_vals[2]), max(bound, max_vals[2])]

        print(f"[RLVisualizer] Rendering {seq_len} frames for mode '{mode_name}'...")
        
        for t in range(seq_len):
            ax.clear()
            
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim) # Y is flipped
            ax.set_zlim(y_lim) # Height (Visual Z)
            ax.set_ylim(z_lim) # Depth (Visual Y)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Z (Depth)')
            ax.set_zlabel('Y (Height)')
            ax.set_title(f"Epoch {epoch} | Frame {t} | Mode: {mode_name}")
            
            # View Init (Front-ish but tilted)
            ax.view_init(elev=10, azim=-90) # azim=-90 puts X axis horizontal, Y axis depth.
            
            # Plot Logic
            for traj in trajectories:
                # Timeline Logic
                # Timeline Logic
                # [Mod] Display all frames logic
                # 原本邏輯會隱藏 inference 前段 (Context)，現在取消隱藏，完整顯示
                # if t < input_n and mode_name == "inference": 
                #      if traj["label"] != "GT":
                #         continue
                
                # Get current pose
                pose = traj["data"][t] # (J, 3)
                
                # Apply Offset (in Data Coordinates)
                pose = pose + traj["offset"]
                
                # Map to Visual Coordinates
                # Plot X = Data X
                # Plot Y = Data Z
                # Plot Z = -Data Y (Flip Y up)
                xs = pose[:, 0]
                ys = pose[:, 2]
                zs = -pose[:, 1]
                
                # Scatter Joints
                ax.scatter(xs, ys, zs, c=traj["color"], s=10, alpha=traj["alpha"])
                
                # Draw Bones
                for (v1, v2) in self.edges:
                    x_pair = [xs[v1], xs[v2]]
                    y_pair = [ys[v1], ys[v2]]
                    z_pair = [zs[v1], zs[v2]]
                    ax.plot(x_pair, y_pair, z_pair, color=traj["color"], 
                            alpha=traj["alpha"], linewidth=traj["linewidth"])

            # Custom Legend
            import matplotlib.lines as mlines
            legend_elements = [
                mlines.Line2D([], [], color='blue', label='GT Input/Target'),
                mlines.Line2D([], [], color='red', label='Pure Prediction'),
                mlines.Line2D([], [], color='darkgreen', alpha=0.7, label='High Reward (Top)'),
                mlines.Line2D([], [], color='lime', alpha=0.5, label='Low Reward (Bot)')
            ]
            ax.legend(handles=legend_elements, loc='upper right')

            # Save frame
            fig.canvas.draw()
            # Compatible with Matplotlib 3.x Agg backend
            s, (width, height) = fig.canvas.print_to_buffer()
            image = np.frombuffer(s, np.uint8).reshape((height, width, 4))
            image = image[:, :, :3] # Drop Alpha channel to get RGB
            frames.append(image)
            
        plt.close(fig)
        
        # Save Video
        save_path = os.path.join(self.output_dir, f"epoch_{epoch}_{mode_name}.mp4")
        imageio.mimsave(save_path, frames, fps=30)
        print(f"[RLVisualizer] Saved video to {save_path}")
