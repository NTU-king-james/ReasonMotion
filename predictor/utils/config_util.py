import argparse
import yaml
import os
import shutil

# 預設 Config 路徑 (可依需求修改)
DEFAULT_CONFIG_PATH = "configs/qwen_rl.yaml"

def load_config(config_path):
    """從 YAML 檔案讀取 config"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def update_config(config, args_dict):
    """根據 CLI 參數覆蓋 config（支援巢狀 key，例如 rl.lr）"""
    for k, v in args_dict.items():
        if v is None:
            continue
        # 跳過非 config 結構的輔助參數 (例如 exp_name, resume_dir)
        if k in ['config', 'resume_dir', 'exp_name']:
            continue
            
        keys = k.split('.')
        d = config
        # 確保路徑存在
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = v
    return config

def save_config(config, save_dir):
    """將 config 儲存到 output 資料夾"""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "config.yaml")
    with open(save_path, 'w') as f:
        yaml.dump(config, f)
    print(f"[Config] Saved to {save_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="RL Training Config")

    # === 基礎控制 ===
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_PATH, help='Path to config yaml file')
    parser.add_argument('--exp_name', type=str, default=None, help='Experiment Name (creates runs/EXP_NAME)')
    parser.add_argument('--resume_dir', type=str, default=None, help="Resume training from folder")
    
    # === SFT Checkpoint ===
    parser.add_argument('--pretrained_ckpt', type=str, help='Path to SFT model checkpoint')

    # === Data ===
    parser.add_argument('--data.data_dir', type=str)
    parser.add_argument('--data.input_n', type=int)
    parser.add_argument('--data.output_n', type=int)
    parser.add_argument('--data.skip_rate', type=int)
    parser.add_argument('--data.mode', type=str)
    parser.add_argument('--data.max_len', type=int)

    # === Model (SFT Architecture) ===
    parser.add_argument('--model.timeemb', type=int)
    parser.add_argument('--model.featureemb', type=int)
    parser.add_argument('--model.textemb', type=int)

    # === Diffusion ===
    parser.add_argument('--diffusion.num_steps', type=int)
    parser.add_argument('--diffusion.schedule', type=str)

    # === RL (GRPO) 參數 [新增] ===
    parser.add_argument('--rl.lr', type=float, help='RL Learning Rate')
    parser.add_argument('--rl.batch_size', type=int)
    parser.add_argument('--rl.num_samples_per_prompt', type=int, help='G (Group Size)')
    parser.add_argument('--rl.diffusion_steps', type=int)
    parser.add_argument('--rl.epsilon', type=float, help='PPO Clip Epsilon')
    parser.add_argument('--rl.kl_coef', type=float, help='KL Penalty Coefficient')
    parser.add_argument('--rl.sampling_std', type=float, help='Exploration Noise Std')
    
    # === Rewards [新增] ===
    parser.add_argument('--rl.w_gt', type=float)
    parser.add_argument('--rl.w_smooth', type=float)
    parser.add_argument('--rl.w_score', type=float)
    parser.add_argument('--rl.fs_reward_ckpt', type=str)
    parser.add_argument('--rl.w_righthand', type=float, help='Sanity Check: Right Hand Height Weight')

    return parser.parse_args()

def get_config():
    """讀取與更新 config"""
    args = parse_args()
    
    # 1. 載入基礎 YAML
    config = load_config(args.config)
    
    # 2. 如果有指定 resume，嘗試載入舊 config (可選)
    # 這裡我們主要依賴 YAML + CLI 覆蓋
    
    # 3. 用 CLI 參數覆蓋
    config = update_config(config, vars(args))
    
    return config, args