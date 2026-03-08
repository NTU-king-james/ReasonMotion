import argparse
import yaml
import os

DEFAULT_CONFIG_PATH = "/home/allen/Diffusion/DePOSit_Skating_Editor_BaseModel/configs/train.yaml"
def load_resume_config(resume_dir):
    """Load config.yaml from resume directory"""
    resume_config_path = os.path.join(resume_dir, "config.yaml")
    if not os.path.exists(resume_config_path):
        raise FileNotFoundError(f"[Resume] config.yaml not found in {resume_dir}")
    return load_config(resume_config_path)

def load_config(config_path):
    """Read config from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def update_config(config, args_dict):
    """Override config based on CLI arguments (supports nested keys)"""
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
    """Save config to output directory"""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "config.yaml")
    with open(save_path, 'w') as f:
        yaml.dump(config, f)
    print(f"[Config] Saved to {save_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Training Config")

    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_PATH, help='Path to config yaml file')
    parser.add_argument('--resume_dir', type=str, default=None, help="Resume training from given output folder (auto load config & checkpoints)")
    # FineFS specific
    parser.add_argument('--data.data_dir', type=str)
    parser.add_argument('--data.output_dir', type=str)
    parser.add_argument('--data.seq_len', type=int)
    parser.add_argument('--data.split', type=int)
    parser.add_argument('--data.data_ratio', type=float)
    parser.add_argument('--data.move_global', type=lambda x: x.lower() == 'true')
    parser.add_argument('--data.all_mask', type=lambda x: x.lower() == 'true')
    parser.add_argument('--data.edit_ratio', type=float)
    parser.add_argument('--data.command_json_path', type=str)
    parser.add_argument('--data.mix_motion', type=lambda x: x.lower() == 'true')
    parser.add_argument('--data.use_magnitude', type=lambda x: x.lower() == 'true')

    # === Training settings ===
    parser.add_argument('--train.epochs', type=int)
    parser.add_argument('--train.batch_size', type=int)
    parser.add_argument('--train.batch_size_test', type=int)
    parser.add_argument('--train.lr', type=float)

    # === Diffusion model settings ===
    parser.add_argument('--diffusion.layers', type=int)
    parser.add_argument('--diffusion.channels', type=int)
    parser.add_argument('--diffusion.nheads', type=int)
    parser.add_argument('--diffusion.diffusion_embedding_dim', type=int)
    parser.add_argument('--diffusion.beta_start', type=float)
    parser.add_argument('--diffusion.beta_end', type=float)
    parser.add_argument('--diffusion.num_steps', type=int)
    parser.add_argument('--diffusion.schedule', type=str, choices=["linear", "cosine"])

    # === Model settings ===
    parser.add_argument('--model.is_unconditional', type=int, choices=[0, 1])
    parser.add_argument('--model.timeemb', type=int)
    parser.add_argument('--model.featureemb', type=int)
    parser.add_argument('--model.text_encoder', type=str, choices=["sentence-transformer", "clip"])
    parser.add_argument('--model.textemb', type=int)
    parser.add_argument('--model.text_mode', type=str, choices=["cross_attn", "concat"])
    parser.add_argument('--model.use_rot_loss', type=lambda x: x.lower() == 'true')

    # === Pre-trained model loading settings ===
    parser.add_argument('--data.model_s', type=str)
    parser.add_argument('--data.model_l', type=str)

    return parser.parse_args()

def get_config():
    """Read and update config, only return, do not save"""
    args = parse_args()
    config = load_config(args.config)
    config = update_config(config, vars(args))
    return config

