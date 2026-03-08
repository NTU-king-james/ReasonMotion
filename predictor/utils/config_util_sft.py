import argparse
import yaml
import os

DEFAULT_CONFIG_PATH = "/home/allen/Diffusion/ReasonMotion_Predictor_BaseModel/configs/train.yaml"
def load_resume_config(resume_dir):
    """從 resume 資料夾載入 config.yaml"""
    resume_config_path = os.path.join(resume_dir, "config.yaml")
    if not os.path.exists(resume_config_path):
        raise FileNotFoundError(f"[Resume] config.yaml not found in {resume_dir}")
    return load_config(resume_config_path)

def load_config(config_path):
    """從 YAML 檔案讀取 config"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def update_config(config, args_dict):
    """根據 CLI 參數覆蓋 config（支援巢狀 key）"""
    for k, v in args_dict.items():
        if v is None:
            continue
        keys = k.split('.')
        d = config
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
    parser = argparse.ArgumentParser(description="Training Config")

    # === 基本設定 ===
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_PATH,
                        help='Path to config yaml file')
    parser.add_argument('--resume_dir', type=str, default=None,
                    help="Resume training from given output folder (auto load config & checkpoints)")
    parser.add_argument('--data.data_dir', type=str, help="Dataset directory")
    parser.add_argument('--data.output_dir', type=str, help="Output directory to save models and logs")

    # === FineFS Dataset 相關 ===
    parser.add_argument('--data.input_n', type=int, help="Number of input frames")
    parser.add_argument('--data.output_n', type=int, help="Number of output frames to predict")
    parser.add_argument('--data.skip_rate', type=int, help="Skip rate for sliding window")
    parser.add_argument('--data.mode', type=str, choices=["full_name", "rotation", "combo_vs_solo"])
    parser.add_argument('--data.max_len', type=int, default=None,
                        help="Maximum sequence length for FineFS dataset (default: None means no limit)")

    # === 訓練設定 ===
    parser.add_argument('--train.epochs', type=int)
    parser.add_argument('--train.batch_size', type=int)
    parser.add_argument('--train.batch_size_test', type=int)
    parser.add_argument('--train.lr', type=float)

    # === Diffusion 模型設定 ===
    parser.add_argument('--diffusion.layers', type=int)
    parser.add_argument('--diffusion.channels', type=int)
    parser.add_argument('--diffusion.nheads', type=int)
    parser.add_argument('--diffusion.diffusion_embedding_dim', type=int)
    parser.add_argument('--diffusion.beta_start', type=float)
    parser.add_argument('--diffusion.beta_end', type=float)
    parser.add_argument('--diffusion.num_steps', type=int)
    parser.add_argument('--diffusion.schedule', type=str, choices=["linear", "cosine"])

    # === 模型設定 ===
    parser.add_argument('--model.is_unconditional', type=int, choices=[0, 1])
    parser.add_argument('--model.timeemb', type=int)
    parser.add_argument('--model.featureemb', type=int)
    parser.add_argument('--model.text_encoder', type=str, choices=["sentence-transformer", "clip"])
    parser.add_argument('--model.textemb', type=int)
    parser.add_argument('--model.text_mode', type=str, choices=["cross_attn", "concat"])
    parser.add_argument('--model.use_rot_loss', type=lambda x: x.lower() == 'true')

    # === Text Injection 設定 ===
    # 支援透過 --text_injection.use_late_injection=True 覆蓋 YAML
    parser.add_argument('--text_injection.use_late_injection', type=lambda x: x.lower() == 'true', help="Injection order")
    parser.add_argument('--text_injection.use_time_aware_film', type=lambda x: x.lower() == 'true', help="FiLM modulation")
    parser.add_argument('--text_injection.decouple_spatial_temporal', type=lambda x: x.lower() == 'true', help="Decouple Attention")
    parser.add_argument('--text_injection.use_clip_encoder', type=lambda x: x.lower() == 'true', help="Use CLIP")
    parser.add_argument('--text_injection.use_deep_text_projector', type=lambda x: x.lower() == 'true', help="Use Deep Projector")

    # === 預訓練模型載入設定 ===
    parser.add_argument('--data.model_s', type=str)
    parser.add_argument('--data.model_l', type=str)

    return parser.parse_args()

def get_config():
    """讀取與更新 config，只回傳，不儲存"""
    args = parse_args()
    config = load_config(args.config)
    config = update_config(config, vars(args))
    return config

