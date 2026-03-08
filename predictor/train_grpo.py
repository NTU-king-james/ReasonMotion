import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import torch
import csv
import shutil
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from datetime import datetime

from model.model import ModelMain
from motion_data.finefs import FineFS
from utils.text_encoder import TextEncoder
from utils.rl_utils import UnifiedRewardModel, GRPOTrainer
from utils.config_util import get_config, save_config


def handle_resume(config, base_dir, device):
    """
    處理 Resume 模式：
    1. 從舊 Run 的 Checkpoint 載入模型權重
    2. 複製舊 Run 的 batch_metrics.csv (截止到指定 batch) 到新 Run
    
    Returns:
        ckpt_path (str): 要載入的 checkpoint 路徑
        resume_batch_offset (int): 新訓練的 batch offset (用於 log 的 Batch ID 接續)
    """
    resume_cfg = config.get('resume', None)
    if not resume_cfg:
        return None, 0
    
    from_run = resume_cfg['from_run']
    from_epoch = resume_cfg.get('from_epoch', 1)
    from_batch = resume_cfg['from_batch']
    
    # 1. 確認 Checkpoint 存在
    ckpt_filename = f"checkpoint_ep{from_epoch}_batch{from_batch}.pth"
    ckpt_path = os.path.join(from_run, "checkpoints", ckpt_filename)
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"❌ Resume checkpoint not found: {ckpt_path}\n"
            f"   Available checkpoints in {os.path.join(from_run, 'checkpoints')}:\n"
            f"   {os.listdir(os.path.join(from_run, 'checkpoints'))[:10]}..."
        )
    
    print(f"🔄 Resume Mode Activated!")
    print(f"   From: {from_run}")
    print(f"   Checkpoint: {ckpt_filename}")
    
    # 2. 複製歷史 Log (截止到指定 batch)
    src_csv = os.path.join(from_run, "loss", "batch_metrics.csv")
    dst_loss_dir = os.path.join(base_dir, "loss")
    dst_csv = os.path.join(dst_loss_dir, "batch_metrics.csv")
    os.makedirs(dst_loss_dir, exist_ok=True)
    
    copied_rows = 0
    if os.path.exists(src_csv):
        with open(src_csv, 'r') as f_in:
            reader = csv.reader(f_in)
            header = next(reader)
            
            rows_to_copy = []
            for row in reader:
                # 解析 Batch ID: E1_B500 -> epoch=1, batch=500
                batch_id = row[0]
                try:
                    import re
                    match = re.match(r'E(\d+)_B(\d+)', batch_id)
                    if match:
                        row_epoch = int(match.group(1))
                        row_batch = int(match.group(2))
                        
                        # 只保留 <= 指定 batch 的記錄
                        if row_epoch < from_epoch or (row_epoch == from_epoch and row_batch <= from_batch):
                            rows_to_copy.append(row)
                except:
                    pass  # 跳過無法解析的行
            
            # 寫入新的 CSV
            with open(dst_csv, 'w', newline='') as f_out:
                writer = csv.writer(f_out)
                writer.writerow(header)
                writer.writerows(rows_to_copy)
            
            copied_rows = len(rows_to_copy)
        
        print(f"   📋 Copied {copied_rows} log entries (up to E{from_epoch}_B{from_batch})")
    else:
        print(f"   ⚠️  No source CSV found at {src_csv}, starting fresh logs")
    
    # 3. 複製舊的 plot 圖片 (可選，方便比較)
    for plot_name in ['batch_metrics_loss.png', 'batch_metrics_rewards.png']:
        src_plot = os.path.join(from_run, "loss", plot_name)
        if os.path.exists(src_plot):
            dst_plot = os.path.join(dst_loss_dir, f"prev_{plot_name}")
            shutil.copy2(src_plot, dst_plot)
    
    return ckpt_path, from_batch


def main():
    config, args = get_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.exp_name:
        exp_name = args.exp_name
    else:
        exp_name = config['project'].get('name', f"exp_{datetime.now().strftime('%m%d_%H%M')}")
    
    base_dir = os.path.join("runs", exp_name)
    ckpt_dir = os.path.join(base_dir, "checkpoints")
    loss_dir = os.path.join(base_dir, "loss")
    os.makedirs(ckpt_dir, exist_ok=True)
    save_config(config, base_dir)
    
    # ===== Resume 處理 =====
    resume_ckpt_path, resume_batch_offset = handle_resume(config, base_dir, device)
    is_resume = resume_ckpt_path is not None
    
    print(f"🚀 Training Start: {exp_name}")
    print(f"📂 Checkpoints: {ckpt_dir}")
    print(f"📊 Logs:        {loss_dir}")
    if is_resume:
        print(f"🔄 Resuming from batch {resume_batch_offset}")

    print("📂 Loading Dataset...")
    data_name = config['data'].get('name', 'FineFS').lower()
    _random_face           = config['data'].get('random_face', False)
    _filter_single_rot     = config['data'].get('filter_single_rotation', False)
    print(f"   Dataset={data_name} random_face={_random_face} filter_single_rotation={_filter_single_rot}")
    
    if data_name == 'h36m':
        from motion_data.h36m import H36M
        dataset = H36M(
            data_dir=config['data']['data_dir'],
            input_n=config['data']['input_n'],
            output_n=config['data']['output_n'],
            skip_rate=config['data']['skip_rate'],
            split=0,
            joints=config['data'].get('joints', 17)
        )
    else:
        dataset = FineFS(
            data_dir=config['data']['data_dir'],
            input_n=config['data']['input_n'],
            output_n=config['data']['output_n'],
            skip_rate=config['data']['skip_rate'],
            mode=config['data']['mode'],
            max_len=config['data']['max_len'],
            random_face=_random_face,
            filter_single_rotation=_filter_single_rot,
            split=0
        )
    dataloader = DataLoader(dataset, batch_size=config['rl']['batch_size'], shuffle=True, num_workers=4, drop_last=True)

    print("🧠 Initializing Model...")
    model = ModelMain(config, device=device, target_dim=config['model']['target_dim'])
    model = model.to(device)

    # 決定要載入哪個 Checkpoint
    if is_resume:
        # Resume 模式：從 RL Checkpoint 載入
        ckpt_path = resume_ckpt_path
        print(f"📥 Loading RL checkpoint (Resume): {ckpt_path}")
    else:
        # 正常模式：從 SFT Checkpoint 載入
        ckpt_path = args.pretrained_ckpt if args.pretrained_ckpt else config.get('pretrained_ckpt')
        print(f"📥 Loading SFT checkpoint: {ckpt_path}")
    
    if ckpt_path and os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location=device)
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)
    else:
        raise FileNotFoundError(f"❌ Checkpoint not found: {ckpt_path}")

    text_encoder = TextEncoder(device=str(device))
    reward_model = UnifiedRewardModel(config=config, device=device)
    trainer = GRPOTrainer(model, reward_model, text_encoder, config, str(device))
    
    # [New] Initialize Visualizer 
    # Must load validation partition (split=1)
    print("🎥 Initializing Visualizer...")
    if data_name == 'h36m':
        val_dataset = H36M(
            data_dir=config['data']['data_dir'],
            input_n=config['data']['input_n'],
            output_n=config['data']['output_n'],
            skip_rate=config['data']['skip_rate'],
            split=1,
            joints=config['data'].get('joints', 17)
        )
    else:
        val_dataset = FineFS(
            data_dir=config['data']['data_dir'],
            input_n=config['data']['input_n'],
            output_n=config['data']['output_n'],
            skip_rate=config['data']['skip_rate'],
            mode=config['data']['mode'],
            max_len=config['data']['max_len'],
            random_face=False,
            filter_single_rotation=_filter_single_rot,
            split=1
        )
    
    from utils.rl_visualizer import RLVisualizer
    visualizer = RLVisualizer(
        output_dir=base_dir,
        val_idx=config['rl'].get('vis_val_idx', 0),
        top_k=config['rl'].get('vis_top_k', 3),
        bot_k=config['rl'].get('vis_bot_k', 3),
        viz_mode=config['rl'].get('viz_mode', 'all'),
        device=device,
        joints=config['model'].get('target_dim', 72) // 3
    )
    visualizer.load_fixed_sample(
        val_dataset, 
        target_path=config['rl'].get('vis_target_path', None)
    )

    # [Check] 記錄初始權重
    initial_weight = model.diffmodel.input_projection.weight.data.clone()
    print(f"📊 Initial Weight Sum: {initial_weight.sum().item():.6f}")

    epochs = 50
    print("🔥 Starting RL Training Loop...")
    
    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch+1}/{epochs} ===")
        
        metrics = trainer.train_epoch(
            dataloader, 
            diffusion_timesteps=config['rl']['diffusion_steps'],
            G=config['rl']['num_samples_per_prompt'],
            # 新增：中間保存與視覺化
            epoch=epoch+1,
            checkpoint_dir=ckpt_dir,
            save_checkpoint_every=config['rl'].get('save_checkpoint_every', None),
            visualize_every=config['rl'].get('visualize_every', None),
            visualizer=visualizer,
            text_encoder=text_encoder,
            current_std=config['rl']['sampling_std'],
            # [New] KL Clipping & Batch-based Ref Update
            max_kl_penalty=config['rl'].get('kl_clip_value', None),
            update_ref_every_batch=config['rl'].get('update_ref_every_batch', None),
            # [New] Resume: Batch offset for log continuity
            batch_offset=resume_batch_offset
        )
        

        
        if metrics:
            print(f"📊 Metrics: R_Tot={metrics['r_total']:.4f} | R_Std={metrics['r_std']:.4f} | R_GT={metrics['r_gt']:.4f} | R_Sm={metrics['r_smooth']:.4f} | KL={metrics['kl']:.4f} | R_score={metrics['r_score']:.4f}")
        else:
            print("📊 Metrics: None (No training steps this epoch?)")
        
        # [Check] 檢查權重是否變化
        current_weight = model.diffmodel.input_projection.weight.data
        diff = (current_weight - initial_weight).abs().sum().item()
        print(f"📉 Weight Change (vs Init): {diff:.6f}")
        if diff == 0:
            print("❌ WARNING: Model weights did NOT change!")

        # Epoch 結束時保存完整 checkpoint
        save_path = os.path.join(ckpt_dir, f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"💾 Saved: {save_path}")
        
        # [New] Run Visualization (Epoch 結束時)
        try:
            visualizer.run_epoch_viz(
                epoch=epoch+1,
                model=model,
                reward_model=reward_model,
                text_encoder=text_encoder,
                current_std=config['rl']['sampling_std'],
                num_variants=config['rl']['num_samples_per_prompt']
            )
        except Exception as e:
            print(f"❌ Visualization Failed: {e}")

if __name__ == "__main__":
    main()