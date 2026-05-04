import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import copy
import numpy as np
from time import time

from utils.fs_reward_utils.fs_reward_model import FSRewardModel
import os
import csv

class BatchLogger:
    """記錄訓練過程中每個 checkpoint 的 metrics"""
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, "batch_metrics.csv")
        # [Mod] Removed meaningless Policy_Loss. Focus on Advantages.
        self.headers = [
            "Batch_ID", "R_Total", "R_Std", "KL_Div", "KL_Penalty",
            "R_GT", "R_Smooth", "R_Score", "R_Rot",
            "R_Total_Window", "R_GT_Window", "R_Smooth_Window",
            "Weighted_GT", "Weighted_Smooth", "Acc_Mean", "Smooth_Floor_Penalty"
        ]
        
        # 初始化 CSV
        file_exists = os.path.exists(self.csv_path)
        if file_exists:
            with open(self.csv_path, 'r', newline='') as f:
                reader = csv.reader(f)
                existing_header = next(reader, None)
            if existing_header != self.headers:
                self.csv_path = os.path.join(log_dir, "batch_metrics_v2.csv")
                file_exists = os.path.exists(self.csv_path)
                print(f"⚠️ Existing batch_metrics.csv has an old header; logging new metrics to {self.csv_path}")

        with open(self.csv_path, 'a' if file_exists else 'w', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(self.headers)
    
    def log_batch(self, batch_id, metrics):
        """記錄單個 batch 的 metrics"""
        row = [
            batch_id,
            f"{metrics.get('r_total', 0):.4f}",
            f"{metrics.get('r_std', 0):.4f}",
            f"{metrics.get('kl', 0):.4f}",
            f"{metrics.get('kl_penalty', 0):.4f}",
            f"{metrics.get('r_gt', 0):.4f}",
            f"{metrics.get('r_smooth', 0):.4f}",
            f"{metrics.get('r_score', 0):.4f}",
            f"{metrics.get('r_rot', 0):.4f}",
            f"{metrics.get('r_total_window', 0):.4f}",
            f"{metrics.get('r_gt_window', 0):.4f}",
            f"{metrics.get('r_smooth_window', 0):.4f}",
            f"{metrics.get('weighted_gt', 0):.4f}",
            f"{metrics.get('weighted_smooth', 0):.4f}",
            f"{metrics.get('acc_mean', 0):.6f}",
            f"{metrics.get('smooth_floor_penalty', 0):.4f}"
        ]
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        
        # 每次记录后更新图表
        self.plot_metrics()
    
    def plot_metrics(self):
        """绘制训练过程中的 metrics 曲线（分两张图）"""
        import pandas as pd
        import matplotlib
        matplotlib.use('Agg')  # 无 GUI 后端
        import matplotlib.pyplot as plt
        
        # 读取 CSV
        if not os.path.exists(self.csv_path):
            return
        
        try:
            df = pd.read_csv(self.csv_path)
            if len(df) == 0:
                return
            
            # [Fix] Calculate Cumulative Batches based on logging interval
            # Assume logging interval is constant, inferred from the first entry (e.g., E1_B500 -> interval 500)
            try:
                import re
                first_batch_cnt = int(re.search(r'B(\d+)', str(df['Batch_ID'].iloc[0])).group(1))
            except:
                first_batch_cnt = 1 # Fallback
            
            x = [i * first_batch_cnt for i in range(1, len(df) + 1)]
            
            # ========== 图1：Training Overview (Total Reward & KL) ==========
            fig1, ax1 = plt.subplots(figsize=(12, 6))
            
            # 1. R_Total (Main Objective)
            ax1.plot(x, df['R_Total'], 'b-', label='Total Reward (Avg)', linewidth=2)
            if 'R_Total_Window' in df.columns:
                ax1.plot(x, df['R_Total_Window'], 'c--', label='Total Reward (Window)', linewidth=1.5)
            
            # 2. R_Std (Diversity)
            # 畫出 R_Total +- R_Std 的陰影區域 (如果 R_Std 存在)
            if 'R_Std' in df.columns:
                r_mean = df['R_Total']
                r_std = df['R_Std']
                ax1.fill_between(x, r_mean - r_std, r_mean + r_std, color='b', alpha=0.1, label='Reward Std (Diversity)')

            # 2.5 KL_Penalty (Same Scale as Reward)
            if 'KL_Penalty' in df.columns:
                ax1.plot(x, df['KL_Penalty'], 'm:', label='KL Penalty (Abs)', linewidth=2)

            ax1.set_xlabel('Total Training Batches', fontsize=12)
            ax1.set_ylabel('Reward Value', color='b', fontsize=12)
            ax1.tick_params(axis='y', labelcolor='b')
            ax1.grid(True, alpha=0.3)
            
            plt.title('Training Progress: Total Reward & KL Divergence', fontsize=14, fontweight='bold')
            
            # 3. KL_Div (Constraint) - Right Axis
            ax2 = ax1.twinx()
            ax2.plot(x, df['KL_Div'], 'r--', label='KL Divergence', linewidth=1.5)
            
            ax2.set_ylabel('KL Divergence', color='r', fontsize=12)
            ax2.tick_params(axis='y', labelcolor='r')
            
            # Combined Legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            plt.tight_layout()
            
            # 保存图1
            loss_plot_path = os.path.join(self.log_dir, 'batch_metrics_loss.png')
            plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig1)
            
            # ========== 图2：Rewards ==========
            fig2, ax3 = plt.subplots(figsize=(12, 6))
            
            ax3.plot(x, df['R_GT'], 'o-', label='R_GT (Ground Truth)', linewidth=2, markersize=4)
            ax3.plot(x, df['R_Smooth'], 's-', label='R_Smooth (Smoothness)', linewidth=2, markersize=4)
            ax3.plot(x, df['R_Score'], '^-', label='R_Score (FS Score)', linewidth=2, markersize=4)
            if 'R_GT_Window' in df.columns:
                ax3.plot(x, df['R_GT_Window'], 'o--', label='R_GT Window', linewidth=1.2, markersize=3, alpha=0.75)
            if 'R_Smooth_Window' in df.columns:
                ax3.plot(x, df['R_Smooth_Window'], 's--', label='R_Smooth Window', linewidth=1.2, markersize=3, alpha=0.75)
            if 'Weighted_GT' in df.columns:
                ax3.plot(x, df['Weighted_GT'], '-', label='Weighted GT', linewidth=1.2, alpha=0.7)
            if 'Weighted_Smooth' in df.columns:
                ax3.plot(x, df['Weighted_Smooth'], '-', label='Weighted Smooth', linewidth=1.2, alpha=0.7)
            
            # 只有当 R_RH 有值时才画
            # 只有當 R_Rot 有值時才畫
            if 'R_Rot' in df.columns and df['R_Rot'].abs().max() > 1e-6:
                ax3.plot(x, df['R_Rot'], '*-', label='R_Rot (Rotation)', linewidth=2, markersize=4)
            
            ax3.set_xlabel('Total Training Batches', fontsize=12)
            ax3.set_ylabel('Reward Value', fontsize=12)
            ax3.grid(True, alpha=0.3)

            if 'Acc_Mean' in df.columns:
                ax4 = ax3.twinx()
                ax4.plot(x, df['Acc_Mean'], 'k:', label='Raw Acc Mean', linewidth=1.5)
                ax4.set_ylabel('Acceleration Mean', color='k', fontsize=12)
                ax4.tick_params(axis='y', labelcolor='k')
                lines3, labels3 = ax3.get_legend_handles_labels()
                lines4, labels4 = ax4.get_legend_handles_labels()
                ax3.legend(lines3 + lines4, labels3 + labels4, loc='best', fontsize=9)
            else:
                ax3.legend(loc='best', fontsize=10)
            
            plt.title('Training Progress: Rewards', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # 保存图2
            reward_plot_path = os.path.join(self.log_dir, 'batch_metrics_rewards.png')
            plt.savefig(reward_plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig2)
            
        except Exception as e:
            print(f"⚠️  绘图失败: {e}")



class UnifiedRewardModel(nn.Module):
    def __init__(self, config=None, device='cuda'):
        super().__init__()
        self.device = device
        
        # 讀取權重配置
        if config is not None and 'rl' in config:
            self.w_gt = float(config['rl'].get('w_gt', 1.0))
            self.w_smooth = float(config['rl'].get('w_smooth', 0.5))
            self.w_score = float(config['rl'].get('w_score', 0.0))
            self.fs_ckpt = config['rl'].get('fs_reward_ckpt', None)
            self.w_rot = float(config['rl'].get('w_rot', 0.0))
            self.rot_threshold = float(config['rl'].get('rot_threshold', 0.1)) # 圈數誤差容忍值 (0.1圈 = 36度)
            self.input_n = int(config.get('data', {}).get('input_n', 0))
            self.smooth_floor = float(config['rl'].get('smooth_floor', 0.0))
            self.smooth_floor_penalty = float(config['rl'].get('smooth_floor_penalty', 0.0))
            print(
                f"🔧 Reward Weights: GT={self.w_gt}, Smooth={self.w_smooth}, "
                f"Score={self.w_score}, Rot={self.w_rot}, "
                f"SmoothFloor={self.smooth_floor}, SmoothPenalty={self.smooth_floor_penalty}"
            )
        else:
            self.w_gt = 1.0
            self.w_smooth = 0.5
            self.w_score = 0.0
            self.fs_ckpt = None
            self.w_rot = 0.0
            self.rot_threshold = 0.1 
            self.input_n = 0
            self.smooth_floor = 0.0
            self.smooth_floor_penalty = 0.0

        # 初始化 FS Reward Model
        self.fs_model = None
        if self.w_score > 0 and self.fs_ckpt:
            print(f"⚖️ Loading Score Model from: {self.fs_ckpt}")
            self.fs_model = FSRewardModel(
                checkpoint_path=self.fs_ckpt,
                device=device,
                scale_type='linear',
                min_score=-5.0,
                max_score=5.0
            )
            self.fs_model.eval()
            for p in self.fs_model.parameters():
                p.requires_grad = False

    def _future_joint_mask(self, mask, samples):
        """Return (B, 1, T, J, 1) mask for future/missing joints."""
        B, _G, T, J, D = samples.shape
        if mask is None:
            future = torch.ones(B, 1, T, J, 1, device=samples.device, dtype=samples.dtype)
            if self.input_n > 0:
                future[:, :, :self.input_n] = 0.0
            return future

        mask = mask.to(samples.device).float()
        if mask.dim() != 3:
            raise ValueError(f"Expected mask with shape (B,T,K) or (B,K,T), got {mask.shape}")

        if mask.shape[1] == T:
            mask_btjd = mask.contiguous().view(B, T, J, D)
        elif mask.shape[2] == T:
            mask_btjd = mask.permute(0, 2, 1).contiguous().view(B, T, J, D)
        else:
            raise ValueError(f"Mask temporal dimension does not match samples: mask={mask.shape}, T={T}")

        observed_joint = (mask_btjd.mean(dim=-1, keepdim=True) > 0.5).float()
        return (1.0 - observed_joint).unsqueeze(1)

    def compute_gt_reward(self, samples, gt, mask=None):
        """L2 Distance Reward (0~1)"""
        diff = samples - gt
        dist_per_joint = torch.norm(diff, dim=-1)
        future_mask = self._future_joint_mask(mask, samples).squeeze(-1)
        denom = future_mask.sum(dim=(-1, -2)).clamp(min=1.0)
        dist = (dist_per_joint * future_mask).sum(dim=(-1, -2)) / denom
        reward_gt = torch.exp(-1.0 * dist)
        # [Range Analysis]
        # Raw: [0, 1] (Exponential negative distance)
        # Weighted (w_gt=0.2): [0, 0.2] -> Very small contribution!
        return reward_gt

    def compute_smoothness_reward(self, samples, mask=None, return_acc=False):
        """Boundary + future acceleration reward (0~1)."""
        if samples.shape[2] < 3:
            reward = torch.ones(samples.shape[0], samples.shape[1], device=samples.device)
            acc_mag = torch.zeros_like(reward)
            return (reward, acc_mag) if return_acc else reward

        vel = torch.diff(samples, dim=2)
        acc = torch.diff(vel, dim=2)
        acc_mag_per_joint = torch.norm(acc, dim=-1)

        future_mask = self._future_joint_mask(mask, samples).squeeze(-1)[:, :, 2:, :]
        if mask is None and self.input_n > 0:
            start = max(self.input_n - 2, 0)
            future_mask = torch.zeros_like(acc_mag_per_joint[:, :1])
            future_mask[:, :, start:, :] = 1.0

        denom = future_mask.sum(dim=(-1, -2)).clamp(min=1.0)
        acc_mag = (acc_mag_per_joint * future_mask).sum(dim=(-1, -2)) / denom
        reward_smooth = torch.exp(-10 * acc_mag)
        # [Range Analysis]
        # Raw: [0, 1] (Exponential negative acceleration)
        # Weighted (w_smooth=1.0): [0, 1.0] -> Balanced.
        return (reward_smooth, acc_mag) if return_acc else reward_smooth

    def compute_score_reward(self, samples):
        """計算動作分數"""
        if self.fs_model is None:
            return torch.zeros(samples.shape[0], samples.shape[1], device=self.device)

        B, G, T, J, D = samples.shape
        samples_flat = samples.reshape(B * G, T, J, D) 
        
        with torch.no_grad():
            scores_flat = self.fs_model(samples_flat) 
            #if torch.rand(1) < 0.01: # Print occasionally
                #print(f"DEBUG: R_Score raw mean: {scores_flat.mean().item():.4f}, min: {scores_flat.min().item():.4f}, max: {scores_flat.max().item():.4f}")
            
        scores = scores_flat.reshape(B, G)
        # [Range Analysis]
        # Raw (FSRewardModel): Typically [-5.0, 5.0] but unbounded.
        # Weighted (w_score=10.0): [-50.0, 50.0] -> DOMINANT TERM!
        # Recommendation: Use Sigmoid or Tanh to squash to [0,1], or normalize.
        return scores

    def compute_righthand_reward(self, samples):
        """右手舉高獎勵"""
        root_y = samples[:, :, :, 0, 1]   # (B, G, T) -1
        rhand_y = samples[:, :, :, 23, 1] # (B, G, T) -2
        
        relative_height = -1*(root_y - rhand_y) #顛倒
        ref_scale = 0.7  # 根據 check_range.py 設定
        score = relative_height / ref_scale
        reward_rh = torch.clamp(score, 0.0, 1.0)
        reward_rh = reward_rh.mean(dim=2) # (B, G)
        # [Range Analysis]
        # Raw: [0, 1] (Clamped)
        # Weighted (w_righthand=0): 0.
        return reward_rh

    def compute_rotation_reward(self, samples, gt_samples):
        """
        計算旋轉圈數獎勵 (Modified: Prefer slightly less rotation)
        目標：生成的總轉圈數 (Total Turns) 落於 [GT - threshold, GT] 區間
        邏輯：
          1. 計算左右髖關節連線向量在 XZ 平面上的角度變化
          2. 累加絕對角度變化量 (Total Angle) -> 轉成圈數 (Total Turns)
          3. 比較 Pred 與 GT 的圈數差 (delta = pred - gt)
          4. 若 -threshold <= delta <= 0 (少轉一點點)，得 1 分
          5. 否則根據偏離該區間的距離進行線性扣分
        """
        # (B, G, T, J, 3)
        # Hip Index: 1(Left), 2(Right) [Check logic: Vis Code uses 1, 2]
        idx_l, idx_r = 1, 2
        
        def get_total_turns(motion):
            # motion: (B, G, T, J, 3) or (B, 1, T, J, 3)
            # Vector: R - L
            vec = motion[..., idx_r, :] - motion[..., idx_l, :] # (..., T, 3)
            
            # [Fix 1] Vector Smoothing (Kernel=5) to remove jitter/glitches
            # 原始數據可能會有單幀跳變 (Outliers)，先對向量做平滑再算角度最穩
            # Reshape for conv1d: (Batch, Channels, Time)
            B_sz, G_sz, T_sz, _ = vec.shape
            vec_flat = vec.view(B_sz * G_sz, T_sz, 3).permute(0, 2, 1) # (N, 3, T)
            
            # AvgPool1d kernel=5, stride=1, padding=2 (Same size output)
            # 注意: Padding 邊緣會補 0，可能導致頭尾數值變小，但對於計算總圈數影響可忽略
            vec_smooth = torch.nn.functional.avg_pool1d(vec_flat, kernel_size=5, stride=1, padding=2, count_include_pad=False)
            
            vec_smooth = vec_smooth.permute(0, 2, 1).view(B_sz, G_sz, T_sz, 3)
            
            # Project to XZ
            vec_x = vec_smooth[..., 0]
            vec_z = vec_smooth[..., 2]
            
            # Atan2 -> (-pi, pi)
            angles = torch.atan2(vec_z, vec_x) # (..., T)
            
            # Delta
            deltas = angles[..., 1:] - angles[..., :-1] # (..., T-1)
            
            # Unwrap: (d + pi) % 2pi - pi
            deltas_unwrapped = (deltas + torch.pi) % (2 * torch.pi) - torch.pi
            
            # [Fix 2] Total Turns = Net Rotation Magnitude (abs of signed sum)
            # 改為計算「淨旋轉量」的絕對值，而非路徑長。
            # 這樣 +1, -1 的抖動會互相抵銷，只保留主要的單向旋轉。
            total_angle = deltas_unwrapped.sum(dim=-1).abs() # (..., )
            
            # Convert to Turns
            return total_angle / (2 * torch.pi)

        turns_pred = get_total_turns(samples)  # (B, G)
        turns_gt = get_total_turns(gt_samples) # (B, 1)
        
        # delta = pred - gt
        # [Mod] Target Range: [GT - 2*threshold, GT - 1*threshold]
        # 這意味著我們希望模型生成的動作比 GT "明顯少轉一點" (例如少 0.1~0.2 圈)
        # 對應 Delta Range: [-2 * threshold, -threshold]
        delta = turns_pred - turns_gt
        
        reward = torch.zeros_like(delta)
        
        lower_bound = -2.0 * self.rot_threshold
        upper_bound = -1.0 * self.rot_threshold
        
        # 1. 滿分區間: [-2*th, -th]
        mask_good = (delta >= lower_bound) & (delta <= upper_bound)
        reward[mask_good] = 1.0
        
        # 2. 扣分區間
        mask_bad = ~mask_good
        bad_delta = delta[mask_bad]
        diff_bad = torch.zeros_like(bad_delta)
        
        # Case A: 轉太多 (delta > upper_bound)
        # e.g. delta = -0.05 (即 GT-0.05), upper = -0.1
        # diff = -0.05 - (-0.1) = 0.05
        mask_over = bad_delta > upper_bound
        diff_bad[mask_over] = bad_delta[mask_over] - upper_bound
        
        # Case B: 轉太少 (delta < lower_bound)
        # e.g. delta = -0.3, lower = -0.2
        # diff = -0.2 - (-0.3) = 0.1
        mask_under = bad_delta < lower_bound
        diff_bad[mask_under] = lower_bound - bad_delta[mask_under]
        
        # [Mod] Asymmetric Decay
        # Case A: Over-rotation (Stricter Penalty) -> k=2.0
        # Case B: Under-rotation (Lenient Penalty) -> k=1.0
        
        decay_val = torch.zeros_like(diff_bad)
        if mask_over.any():
            decay_val[mask_over] = 2.0 * diff_bad[mask_over]
        
        if mask_under.any():
            decay_val[mask_under] = 1.0 * diff_bad[mask_under]
        
        # [Fix] decay_val is already 1D (masked), no need to index it again
        reward[mask_bad] = 1.0 - decay_val
        reward = torch.clamp(reward, min=0.0)
        
        return reward

    def forward(self, samples, gt_motion, text_emb=None, mask=None):
        B, G, T, J, D = samples.shape
        
        if gt_motion.dim() == 4:
            gt_expanded = gt_motion.unsqueeze(1)
        else:
            gt_expanded = gt_motion.view(B, 1, T, J, D)

        r_gt = self.compute_gt_reward(samples, gt_expanded, mask=mask)
        r_smooth, acc_mag = self.compute_smoothness_reward(samples, mask=mask, return_acc=True)
        r_score = self.compute_score_reward(samples) if self.w_score > 0 else torch.zeros_like(r_gt)

        r_rot = self.compute_rotation_reward(samples, gt_expanded) if self.w_rot > 0 else torch.zeros_like(r_gt)

        smooth_floor_penalty = torch.relu(self.smooth_floor - r_smooth) * self.smooth_floor_penalty
        
        total_reward = (
            self.w_gt * r_gt + 
            self.w_smooth * r_smooth +
            self.w_score * r_score +
            self.w_rot * r_rot -
            smooth_floor_penalty
        )
        # [Total Reward Analysis]
        # Current Range: Approx [-50, 50] + [0, 1.2]
        # The total reward is completely dominated by r_score.
        # w_gt (0.2) is likely noise compared to score variance.
        
        metrics = {
            "r_gt": r_gt.mean().item(),
            "r_smooth": r_smooth.mean().item(),
            "r_score": r_score.mean().item(),
            "r_rot": r_rot.mean().item(),
            "weighted_gt": (self.w_gt * r_gt).mean().item(),
            "weighted_smooth": (self.w_smooth * r_smooth).mean().item(),
            "acc_mean": acc_mag.mean().item(),
            "smooth_floor_penalty": smooth_floor_penalty.mean().item(),
            "total": total_reward.mean().item()
        }
        
        return total_reward, metrics


class GRPOTrainer:
    def __init__(self, model, reward_model, text_encoder, config=None, device='cuda'):
        self.device = device
        self.model = model.to(device)
        self.reward_model = reward_model.to(device)
        self.text_encoder = text_encoder.to(device)
        self.config = config

        if config and 'rl' in config:
            self.lr = float(config['rl'].get('lr', 5e-6))
            self.epsilon = float(config['rl'].get('epsilon', 0.2))
            self.kl_coef = float(config['rl'].get('kl_coef', 0.05))
            self.ppo_epochs = int(config['rl'].get('ppo_epochs', 2))
            self.max_train_batches = config['rl'].get('max_train_batches', None)
        else:
            self.lr = 5e-6
            self.epsilon = 0.2
            self.kl_coef = 0.05
            self.ppo_epochs = 2
            self.max_train_batches = None

        if self.max_train_batches is not None:
            self.max_train_batches = int(self.max_train_batches)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        # [New] LR Scheduler Setup
        self.scheduler = None
        if config and 'rl' in config and 'lr_schedule' in config['rl']:
            sched_conf = config['rl']['lr_schedule']
            sched_type = sched_conf.get('type', 'constant')
            
            if sched_type == 'step':
                step_size = sched_conf.get('step_size', 5000)
                gamma = sched_conf.get('gamma', 0.5)
                self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
                print(f"📉 LR Scheduler: StepLR (step_size={step_size}, gamma={gamma})")
            elif sched_type == 'cosine':
                t_max = sched_conf.get('t_max', 30000)
                eta_min = sched_conf.get('eta_min', 1e-7)
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=t_max, eta_min=eta_min)
                print(f"📉 LR Scheduler: CosineAnnealingLR (T_max={t_max}, eta_min={eta_min})")
            else:
                print(f"ℹ️ LR Scheduler: Constant (No scheduler used)")
        
        # 初始化 Batch Logger（記錄中間 checkpoint 的 metrics）
        log_dir = config['project']['output_dir'] + '/loss' if config else './logs'
        self.batch_logger = BatchLogger(log_dir)
        
        # [Check] 確保模型有可訓練參數
        print(f"🔧 Optimizer checking...")
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.model.parameters())
        print(f"   Trainable params: {trainable_params} / {all_params}")
        if trainable_params == 0:
            raise ValueError("❌ Fatal Error: No trainable parameters found! Check requires_grad.")

        # [Check] Deepcopy 模型並凍結
        if isinstance(self.model, nn.DataParallel):
            self.old_model = copy.deepcopy(self.model.module)
            self.old_model = nn.DataParallel(self.old_model).to(device)
            self.ref_model = copy.deepcopy(self.model.module)
            self.ref_model = nn.DataParallel(self.ref_model).to(device)
        else:
            self.old_model = copy.deepcopy(self.model).to(device)
            self.ref_model = copy.deepcopy(self.model).to(device)

        self.old_model.eval()
        self.ref_model.eval()

        for p in self.old_model.parameters(): p.requires_grad = False
        for p in self.ref_model.parameters(): p.requires_grad = False

        self._load_fixed_ref_checkpoint()
        
        self.just_updated_ref = False

    def _unwrap_model(self, model):
        return model.module if isinstance(model, nn.DataParallel) else model

    def _copy_model_state(self, src, dst):
        self._unwrap_model(dst).load_state_dict(self._unwrap_model(src).state_dict())

    def update_old_model(self):
        self._copy_model_state(self.model, self.old_model)
        self.old_model.eval()

    def _load_fixed_ref_checkpoint(self):
        if not self.config:
            return

        ref_ckpt = self.config.get('rl', {}).get('ref_ckpt') or self.config.get('pretrained_ckpt')
        if not ref_ckpt:
            return
        if not os.path.exists(ref_ckpt):
            print(f"⚠️ Fixed reference checkpoint not found, keeping initial ref_model: {ref_ckpt}")
            return

        print(f"📌 Loading fixed KL reference: {ref_ckpt}")
        state_dict = torch.load(ref_ckpt, map_location=self.device)
        if isinstance(state_dict, dict):
            for key in ("state_dict", "model_state_dict", "model"):
                if key in state_dict and isinstance(state_dict[key], dict):
                    state_dict = state_dict[key]
                    break

        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        target = self._unwrap_model(self.ref_model)
        target.load_state_dict(new_state_dict, strict=False)
        self.ref_model.eval()

    @staticmethod
    def _approx_kl_from_logps(policy_step_logps, ref_step_logps):
        log_ref_policy = torch.clamp(ref_step_logps - policy_step_logps, min=-20.0, max=20.0)
        return torch.exp(log_ref_policy) - log_ref_policy - 1.0

    def train_epoch(self, train_loader, diffusion_timesteps=50, G=16, 
                    epoch=None, checkpoint_dir=None, 
                    save_checkpoint_every=None, visualize_every=None,
                    visualizer=None, text_encoder=None, current_std=0.02,
                    max_kl_penalty=None, update_ref_every_batch=None,
                    batch_offset=0):
        """
        訓練一個 Epoch
        
        新增參數：
        - epoch: 當前 epoch 編號 (用於保存 checkpoint 命名)
        - checkpoint_dir: checkpoint 保存目錄
        - save_checkpoint_every: 每 N 個 batch 保存一次 checkpoint
        - visualize_every: 每 N 個 batch 執行一次視覺化
        - visualizer: RLVisualizer 實例
        - text_encoder: TextEncoder 實例
        - current_std: 當前採樣標準差
        """
        # [Critical Fix] Stick to Eval mode for RL!
        # This prevents Dropout randomness from causing Negative KL.
        # Gradients will still flow because backprop_trajectory_loss handles it manually.
        self.model.eval()
        
        metric_keys = [
            "kl_penalty", "kl", "r_total", "r_std",
            "r_gt", "r_smooth", "r_score", "r_rot",
            "weighted_gt", "weighted_smooth", "acc_mean", "smooth_floor_penalty"
        ]
        epoch_metrics = {key: [] for key in metric_keys}
        window_metrics = {key: [] for key in metric_keys}
        
        progress = tqdm(train_loader, desc="RL Training")
        if update_ref_every_batch is not None:
            print("📌 Fixed reference mode: ignoring update_ref_every_batch; KL ref stays at base checkpoint.")
        
        for batch_idx, batch in enumerate(progress):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            model_engine = self._unwrap_model(self.model)
            old_engine = self._unwrap_model(self.old_model)
            ref_engine = self._unwrap_model(self.ref_model)

            text_emb = self.text_encoder(batch["motion_name"])
            self.update_old_model()
            for engine in (model_engine, old_engine, ref_engine):
                if hasattr(engine, "sampling_std"):
                    engine.sampling_std = float(current_std)

            with torch.no_grad():
                # 1. Rollout from the frozen old policy for PPO ratios.
                final_samples, _total_log_probs, old_step_log_probs, all_latents, all_routing = old_engine.sample_trajectory(
                    text_emb, batch, G, return_step_log_probs=True
                )
                old_step_log_probs = old_step_log_probs.detach()

                # 2. Reward Calculation
                # [Fix] 必須將 Epsilon 轉回 x0 (Action) 才能算 Reward -> sample_trajectory 已經回傳 x0 (final_samples)
                # final_samples: (B, G, K, L). K=J*3, L=T.
                
                # Reshape for Reward Model: (B, G, T, J, 3)
                B, G, K, L = final_samples.shape
                J = K // 3
                # final_samples is (B, G, J*3, T). Permute to (B, G, T, J, 3).
                final_samples_reshaped = final_samples.permute(0, 1, 3, 2).reshape(B, G, L, J, 3)
                
                # Pose GT: (B, T, J, 3)
                pose_gt = batch["pose"].to(self.device).float()
                if pose_gt.dim() == 3:
                    pose_gt = pose_gt.view(B, L, J, 3) # Assuming T=L
                
                rewards, metrics = self.reward_model(final_samples_reshaped, pose_gt, mask=batch.get("mask"))
                
                # 3. Fixed-reference KL and group-relative advantage.
                ref_step_log_probs = ref_engine.get_trajectory_step_log_probs(all_latents, text_emb, batch).detach()
                approx_kl = self._approx_kl_from_logps(old_step_log_probs, ref_step_log_probs)
                kl_penalty = self.kl_coef * approx_kl
                if max_kl_penalty is not None:
                    kl_penalty = torch.clamp(kl_penalty, max=max_kl_penalty)

                mean_r = rewards.mean(dim=1, keepdim=True)
                std_r = rewards.std(dim=1, keepdim=True, unbiased=False)
                advantages_task = (rewards - mean_r) / (std_r + 1e-8)
                advantages = advantages_task.detach()

                kl_div = approx_kl.mean()
                r_total_mean = rewards.mean().item()
                r_std_mean = rewards.std(unbiased=False).item() # Whole batch std (approx diversity)

            # 4. PPO updates on the same detached rollout.
            for ppo_epoch in range(self.ppo_epochs):
                self.optimizer.zero_grad()
                model_engine.backprop_trajectory_loss(
                    all_latents,
                    text_emb,
                    batch,
                    advantages,
                    old_step_log_probs=old_step_log_probs,
                    ref_step_log_probs=ref_step_log_probs,
                    epsilon=self.epsilon,
                    kl_coef=self.kl_coef,
                    max_kl_penalty=max_kl_penalty,
                    all_routing=all_routing,
                )
                
                # [Debug] 梯度檢查
                if batch_idx % 50 == 0 and ppo_epoch == 0:
                    grad_norm = 0.0
                    has_grad = False
                    for name, p in self.model.named_parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            grad_norm += param_norm.item() ** 2
                            has_grad = True
                    grad_norm = grad_norm ** 0.5
                    
                    if not has_grad or grad_norm == 0:
                        print("   🔴 CRITICAL: Gradient is ZERO! Check connection graph.")
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            # [New] Step Scheduler: keep one scheduler step per rollout batch.
            if self.scheduler:
                self.scheduler.step()

            # Logging ...
            # epoch_metrics["policy_loss"].append(loss.item())
            # epoch_metrics["adv_task"].append(advantages_task.mean().item()) # Always 0
            batch_metrics = {
                "kl_penalty": kl_penalty.mean().item(),
                "kl": kl_div.item(),
                "r_total": r_total_mean,
                "r_std": r_std_mean,
                "r_gt": metrics["r_gt"],
                "r_smooth": metrics["r_smooth"],
                "r_score": metrics.get("r_score", 0.0),
                "r_rot": metrics.get("r_rot", 0.0),
                "weighted_gt": metrics.get("weighted_gt", 0.0),
                "weighted_smooth": metrics.get("weighted_smooth", 0.0),
                "acc_mean": metrics.get("acc_mean", 0.0),
                "smooth_floor_penalty": metrics.get("smooth_floor_penalty", 0.0),
            }
            for key, value in batch_metrics.items():
                epoch_metrics[key].append(value)
                window_metrics[key].append(value)

            if batch_idx % 5 == 0:
                # [Mod] Dynamic Tqdm Postfix based on Reward Weights
                # Show Total Reward (Real Performance) instead of Adv
                postfix_dict = {
                    "Tot": f"{r_total_mean:.3f}", 
                    "Std": f"{r_std_mean:.3f}",
                    "Acc": f"{metrics.get('acc_mean', 0.0):.4f}"
                }
                if metrics.get("smooth_floor_penalty", 0.0) > 0:
                    postfix_dict["SmPen"] = f"{metrics['smooth_floor_penalty']:.3f}"
                
                # Definite metrics
                metrics_map = {
                    "GT": (self.reward_model.w_gt, f"{metrics['r_gt']:.3f}"),
                    "Sm": (self.reward_model.w_smooth, f"{metrics['r_smooth']:.3f}"),
                    "Sc": (self.reward_model.w_score, f"{metrics.get('r_score', 0.0):.3f}"),
                    "Rot": (self.reward_model.w_rot, f"{metrics.get('r_rot', 0.0):.3f}")
                }
                
                # Filter and Sort by Weight (Descending)
                active_metrics = []
                for name, (w, val_str) in metrics_map.items():
                    if abs(w) > 1e-6: # Show if weight is non-zero
                        active_metrics.append((w, name, val_str))
                
                # Sort: primary key = weight (desc), secondary = name
                active_metrics.sort(key=lambda x: x[0], reverse=True)
                
                # Add to dict
                for _, name, val_str in active_metrics:
                    postfix_dict[name] = val_str
                    
                # Always add KL at the end
                postfix_dict["KL"] = f"{kl_div.item():.3f}"
                
                progress.set_postfix(postfix_dict)
            
            # Fixed-reference KL: do not periodically move ref_model toward the
            # current policy, otherwise drift becomes the new baseline.

            # ===== Checkpoint Saving =====
            if save_checkpoint_every is not None and (batch_idx + 1) % save_checkpoint_every == 0:
                actual_batch = batch_idx + 1 + batch_offset
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_ep{epoch}_batch{actual_batch}.pth")
                
                # 保存模型
                if isinstance(self.model, nn.DataParallel):
                    torch.save(self.model.module.state_dict(), checkpoint_path)
                else:
                    torch.save(self.model.state_dict(), checkpoint_path)
                
                # 計算當前平均 metrics (For Print)
                current_avg_metrics = {k: np.mean(v) if v else 0.0 for k, v in epoch_metrics.items()}
                
                print(f"\n💾 [Batch {actual_batch}] Checkpoint saved: {os.path.basename(checkpoint_path)}")
                print(f"   📊 Avg Metrics: R_Tot={current_avg_metrics['r_total']:.4f} (±{current_avg_metrics['r_std']:.4f}) | "
                      f"R_GT={current_avg_metrics['r_gt']:.4f} | R_Sm={current_avg_metrics['r_smooth']:.4f} | "
                      f"R_Sc={current_avg_metrics['r_score']:.4f} | KL={current_avg_metrics['kl']:.4f}")

            # ===== Metrics Logging (CSV) =====
            # Log 100x more frequently than checkpoint saving
            log_every = max(1, save_checkpoint_every // 10) if save_checkpoint_every else 100
            
            if (batch_idx + 1) % log_every == 0:
                current_avg_metrics = {k: np.mean(v) if v else 0.0 for k, v in epoch_metrics.items()}
                window_avg_metrics = {k: np.mean(v) if v else 0.0 for k, v in window_metrics.items()}
                # Keep running-mean metrics, but log KL terms as current batch values.
                current_log_metrics = dict(current_avg_metrics)
                if epoch_metrics["kl"]:
                    current_log_metrics["kl"] = epoch_metrics["kl"][-1]
                if epoch_metrics["kl_penalty"]:
                    current_log_metrics["kl_penalty"] = epoch_metrics["kl_penalty"][-1]
                current_log_metrics["r_total_window"] = window_avg_metrics["r_total"]
                current_log_metrics["r_gt_window"] = window_avg_metrics["r_gt"]
                current_log_metrics["r_smooth_window"] = window_avg_metrics["r_smooth"]
                current_log_metrics["weighted_gt"] = window_avg_metrics["weighted_gt"]
                current_log_metrics["weighted_smooth"] = window_avg_metrics["weighted_smooth"]
                current_log_metrics["acc_mean"] = window_avg_metrics["acc_mean"]
                current_log_metrics["smooth_floor_penalty"] = window_avg_metrics["smooth_floor_penalty"]
                
                if hasattr(self, 'batch_logger'):
                    actual_batch = batch_idx + 1 + batch_offset
                    batch_id = f"E{epoch}_B{actual_batch}"
                    self.batch_logger.log_batch(batch_id, current_log_metrics)
                    for key in window_metrics:
                        window_metrics[key] = []
                    
                    # [Mod] Clear metrics ONLY if Ref Model was just updated.
                    # This resets the accumulation cycle (e.g., at Batch 3000).
                    if self.just_updated_ref:
                        for key in epoch_metrics:
                            epoch_metrics[key] = []
                        self.just_updated_ref = False


            
            # ===== 中間視覺化 =====
            if visualize_every is not None and (batch_idx + 1) % visualize_every == 0:
                if visualizer is not None and text_encoder is not None:
                    actual_batch_viz = batch_idx + 1 + batch_offset
                    print(f"\n🎥 [Batch {actual_batch_viz}] Running visualization...")
                    try:
                        # 都在 eval 模式下，無需切換
                        visualizer.run_epoch_viz(
                            epoch=f"{epoch}_batch{actual_batch_viz}",  # 特殊命名以區分中間視覺化
                            model=self.model,
                            reward_model=self.reward_model,
                            text_encoder=text_encoder,
                            current_std=current_std,
                            num_variants=G
                        )
                        # 切換回訓練模式
                        # 保持 eval 模式
                        print(f"   ✅ Visualization completed.")
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        print(f"   ❌ Visualization failed: {e}")
                        # self.model.train()  # 確保恢復訓練模式 -> Keep Eval

            if self.max_train_batches is not None and (batch_idx + 1) >= self.max_train_batches:
                print(f"\n⏹️ Reached max_train_batches={self.max_train_batches}; ending this short RL run.")
                break

        avg_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()} if epoch_metrics["r_total"] else {}
        return avg_metrics

    def update_ref_model(self):
        if isinstance(self.model, nn.DataParallel):
            self.ref_model.module.load_state_dict(self.model.module.state_dict())
        else:
            self.ref_model.load_state_dict(self.model.state_dict())
