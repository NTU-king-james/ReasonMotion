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
        self.headers = ["Batch_ID", "Policy_Loss", "GRPO_Loss", "KL_Div", "R_GT", "R_Smooth", "R_Score", "R_RH"]
        
        # 初始化 CSV
        file_exists = os.path.exists(self.csv_path)
        with open(self.csv_path, 'a' if file_exists else 'w', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(self.headers)
    
    def log_batch(self, batch_id, metrics):
        """記錄單個 batch 的 metrics"""
        row = [
            batch_id,
            f"{metrics.get('policy_loss', 0):.4f}",
            f"{metrics.get('grpo_loss', 0):.4f}",
            f"{metrics.get('kl', 0):.4f}",
            f"{metrics.get('r_gt', 0):.4f}",
            f"{metrics.get('r_smooth', 0):.4f}",
            f"{metrics.get('r_score', 0):.4f}",
            f"{metrics.get('r_rh', 0):.4f}"
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
            
            # ========== 图1：Loss 和 KL_Div ==========
            fig1, ax1 = plt.subplots(figsize=(12, 6))
            
            # 绘制 Loss（左侧 y 轴）
            ax1.plot(x, df['Policy_Loss'], 'b-', label='Policy Loss', linewidth=2)
            ax1.plot(x, df['GRPO_Loss'], 'g--', label='GRPO Loss', linewidth=1.5)
            ax1.set_xlabel('Total Training Batches', fontsize=12)
            ax1.set_ylabel('Loss', color='b', fontsize=12)
            ax1.tick_params(axis='y', labelcolor='b')
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3)
            
            # 绘制 KL_Div（右侧 y 轴，因为 scale 可能差很多）
            ax2 = ax1.twinx()
            ax2.plot(x, df['KL_Div'], 'r-', label='KL Divergence', linewidth=2, alpha=0.7)
            ax2.set_ylabel('KL Divergence', color='r', fontsize=12)
            ax2.tick_params(axis='y', labelcolor='r')
            ax2.legend(loc='upper right')
            
            plt.title('Training Progress: Loss & KL Divergence', fontsize=14, fontweight='bold')
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
            
            # 只有当 R_RH 有值时才画
            if df['R_RH'].abs().max() > 1e-6:
                ax3.plot(x, df['R_RH'], 'd-', label='R_RH (Right Hand)', linewidth=2, markersize=4)
            
            ax3.set_xlabel('Total Training Batches', fontsize=12)
            ax3.set_ylabel('Reward Value', fontsize=12)
            ax3.legend(loc='best', fontsize=10)
            ax3.grid(True, alpha=0.3)
            
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
            self.w_righthand = float(config['rl'].get('w_righthand', 0.0))
            print(f"🔧 Reward Weights: GT={self.w_gt}, Smooth={self.w_smooth}, Score={self.w_score}, RH={self.w_righthand}")
        else:
            self.w_gt = 1.0
            self.w_smooth = 0.5
            self.w_score = 0.0
            self.fs_ckpt = None
            self.w_righthand = 0.0 

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

    def compute_gt_reward(self, samples, gt):
        """L2 Distance Reward (0~1)"""
        diff = samples - gt
        dist = torch.norm(diff, dim=-1).mean(dim=(-1, -2))
        reward_gt = torch.exp(-1.0 * dist)
        return reward_gt

    def compute_smoothness_reward(self, samples):
        """Smoothness Reward (0~1)"""
        vel = torch.diff(samples, dim=2) 
        acc = torch.diff(vel, dim=2)
        acc_mag = torch.norm(acc, dim=-1).mean(dim=(-1, -2))
        reward_smooth = torch.exp(-10 * acc_mag)
        return reward_smooth

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
        return reward_rh

    def forward(self, samples, gt_motion, text_emb=None):
        B, G, T, J, D = samples.shape
        
        if gt_motion.dim() == 4:
            gt_expanded = gt_motion.unsqueeze(1)
        else:
            gt_expanded = gt_motion.view(B, 1, T, J, D)

        r_gt = self.compute_gt_reward(samples, gt_expanded)
        r_smooth = self.compute_smoothness_reward(samples)
        r_score = self.compute_score_reward(samples) if self.w_score > 0 else torch.zeros_like(r_gt)
        r_rh = self.compute_righthand_reward(samples) if self.w_righthand > 0 else torch.zeros_like(r_gt)
        
        total_reward = (
            self.w_gt * r_gt + 
            self.w_smooth * r_smooth +
            self.w_score * r_score +
            self.w_righthand * r_rh 
        )
        
        metrics = {
            "r_gt": r_gt.mean().item(),
            "r_smooth": r_smooth.mean().item(),
            "r_score": r_score.mean().item(),
            "r_rh": r_rh.mean().item(),
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
        else:
            self.lr = 5e-6
            self.epsilon = 0.2
            self.kl_coef = 0.05

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

    def train_epoch(self, train_loader, diffusion_timesteps=50, G=16, 
                    epoch=None, checkpoint_dir=None, 
                    save_checkpoint_every=None, visualize_every=None,
                    visualizer=None, text_encoder=None, current_std=0.02,
                    max_kl_penalty=None, update_ref_every_batch=None):
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
        
        epoch_metrics = {
            "policy_loss": [], "grpo_loss": [], "kl": [], 
            "r_gt": [], "r_smooth": [], "r_score": [], "r_rh": [] 
        }
        
        progress = tqdm(train_loader, desc="RL Training")
        
        for batch_idx, batch in enumerate(progress):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # ... (模型引擎獲取代碼保持不變) ...
            model_engine = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
            old_engine = self.old_model.module if isinstance(self.old_model, nn.DataParallel) else self.old_model
            ref_engine = self.ref_model.module if isinstance(self.ref_model, nn.DataParallel) else self.ref_model

            text_emb = self.text_encoder(batch["motion_name"])

            # 1. Trajectory Generation (Pass 1: Inference / No Grad)
            with torch.no_grad():
                final_samples, total_log_probs, all_latents = model_engine.sample_trajectory(text_emb, batch, G)

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
                
                rewards, metrics = self.reward_model(final_samples_reshaped, pose_gt)
                
                # 3. Policy Loss Prep (KL & Advantage)
                log_prob_old = total_log_probs
                
                # Calculate Policy Log Prob (Already in Eval Mode)
                log_prob_old_eval = model_engine.get_trajectory_log_prob(all_latents, text_emb, batch)

                # Ref Model is already in Eval Mode
                log_prob_ref = ref_engine.get_trajectory_log_prob(all_latents, text_emb, batch)
                
                # KL = Policy(Eval) - Ref(Eval)
                # This ensures KL is close to 0 initially
                kl = log_prob_old_eval - log_prob_ref # (B, G)
                
                # [Mod] KL Punishment with Hard Clipping
                # 計算原始懲罰值
                kl_penalty = self.kl_coef * kl
                
                # 如果設定了 max_kl_penalty，則將懲罰值限制在該範圍內
                # 例如 max_kl_penalty=5.0，則即使 KL=200 (penalty=10)，也只扣 5 分
                # 這能防止 KL 過大時懲罰項完全蓋過 Reward
                if max_kl_penalty is not None:
                    kl_penalty = torch.clamp(kl_penalty, max=max_kl_penalty)

                # Adjust Reward with KL Constraint (PPO-style: R_total = R - beta * KL)
                rewards_adjusted = rewards - kl_penalty
                
                # Use adjusted rewards for Advantage
                mean_r = rewards_adjusted.mean(dim=1, keepdim=True)
                std_r = rewards_adjusted.std(dim=1, keepdim=True)
                advantages = (rewards_adjusted - mean_r) / (std_r + 1e-8)
                
                # Logging Metrics (Approx)
                kl_div = kl.mean()
                grpo_loss = -(advantages * log_prob_old).mean() # Approx
                loss = grpo_loss + kl_penalty.mean() # Log actual penalty used

            # 4. Backward (Pass 2: Training / Grad)
            # Memory Efficient: Re-compute gradients stepwise
            self.optimizer.zero_grad()
            model_engine.backprop_trajectory_loss(all_latents, text_emb, batch, advantages)
            
            # [Debug] 梯度檢查
            if batch_idx % 100 == 0:
                grad_norm = 0.0
                has_grad = False
                for name, p in self.model.named_parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        grad_norm += param_norm.item() ** 2
                        has_grad = True
                grad_norm = grad_norm ** 0.5
                #print(f"   [Step {batch_idx}] Grad Norm: {grad_norm:.6f}")
                
                if not has_grad or grad_norm == 0:
                    print("   🔴 CRITICAL: Gradient is ZERO! Check connection graph.")
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # [New] Step Scheduler
            if self.scheduler:
                self.scheduler.step()

            # Logging ... (保持不變)
            epoch_metrics["policy_loss"].append(loss.item())
            epoch_metrics["grpo_loss"].append(grpo_loss.item())
            epoch_metrics["kl"].append(kl_div.item())
            epoch_metrics["r_gt"].append(metrics["r_gt"])
            epoch_metrics["r_smooth"].append(metrics["r_smooth"])
            epoch_metrics["r_score"].append(metrics.get("r_score", 0.0))
            epoch_metrics["r_rh"].append(metrics.get("r_rh", 0.0))

            if batch_idx % 5 == 0:
                # [Mod] Dynamic Tqdm Postfix based on Reward Weights
                # Show Net Advantage (Push/Pull) instead of raw Loss
                postfix_dict = {"Adv": f"{advantages.mean().item():.3f}"}
                
                # Definite metrics
                metrics_map = {
                    "R_GT": (self.reward_model.w_gt, f"{metrics['r_gt']:.4f}"),
                    "R_Sm": (self.reward_model.w_smooth, f"{metrics['r_smooth']:.4f}"),
                    "R_Sc": (self.reward_model.w_score, f"{metrics.get('r_score', 0.0):.4f}"),
                    "R_RH": (self.reward_model.w_righthand, f"{metrics.get('r_rh', 0.0):.4f}")
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
                postfix_dict["KL"] = f"{kl_div.item():.4f}"
                
                progress.set_postfix(postfix_dict)
            
            # ===== [New] Reference Model Periodic Update (Batch-based) =====
            # 根據 batch 數進行 Reference Model 的更新
            # 這能定期重置 KL 散度，讓模型基於新的能力繼續優化，避免 KL 累積過大導致崩潰
            # 類似於 Curriculum Learning 或 Iterative RL
            if update_ref_every_batch is not None and (batch_idx + 1) % update_ref_every_batch == 0:
                self.update_ref_model()
                print(f"\n🔄 [Batch {batch_idx+1}] Reference model updated! KL Reset.")

            # ===== 中間 Checkpoint 保存 =====
            if save_checkpoint_every is not None and (batch_idx + 1) % save_checkpoint_every == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_ep{epoch}_batch{batch_idx+1}.pth")
                
                # 保存模型
                if isinstance(self.model, nn.DataParallel):
                    torch.save(self.model.module.state_dict(), checkpoint_path)
                else:
                    torch.save(self.model.state_dict(), checkpoint_path)
                
                # 計算當前平均 metrics
                current_avg_metrics = {k: np.mean(v) if v else 0.0 for k, v in epoch_metrics.items()}
                
                print(f"\n💾 [Batch {batch_idx+1}] Checkpoint saved: {os.path.basename(checkpoint_path)}")
                print(f"   📊 Avg Metrics: Adv={current_avg_metrics['adv_total']:.4f} | "
                      f"R_GT={current_avg_metrics['r_gt']:.4f} | R_Sm={current_avg_metrics['r_smooth']:.4f} | "
                      f"R_Sc={current_avg_metrics['r_score']:.4f} | KL={current_avg_metrics['kl']:.4f}")
                
                # 記錄到 CSV (使用 batch 作為標識)
                if hasattr(self, 'batch_logger'):
                    batch_id = f"E{epoch}_B{batch_idx+1}"
                    self.batch_logger.log_batch(batch_id, current_avg_metrics)
            
            # ===== 中間視覺化 =====
            if visualize_every is not None and (batch_idx + 1) % visualize_every == 0:
                if visualizer is not None and text_encoder is not None:
                    print(f"\n🎥 [Batch {batch_idx+1}] Running visualization...")
                    try:
                        # 都在 eval 模式下，無需切換
                        visualizer.run_epoch_viz(
                            epoch=f"{epoch}_batch{batch_idx+1}",  # 特殊命名以區分中間視覺化
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

        avg_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()} if epoch_metrics["adv_total"] else {}
        return avg_metrics

    def update_ref_model(self):
        if isinstance(self.model, nn.DataParallel):
            self.ref_model.module.load_state_dict(self.model.module.state_dict())
        else:
            self.ref_model.load_state_dict(self.model.state_dict())