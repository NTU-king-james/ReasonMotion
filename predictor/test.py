"""
CUDA_VISIBLE_DEVICES=1 python test.py --ckpt /home/allen/Diffusion/ReasonMotion_SFT_GRPO/runs/0102_SFT_RL_FineFS_vis_01/checkpoints/model_epoch_50.pth

Explanation:
1. --ckpt: The path to the model checkpoint file (.pth) to evaluate.
2. --config: (Optional) Path to the YAML config file.
   - Why it's optional: The script automatically tries to find 'config.yaml' in the *parent directory* of the checkpoint (assuming standard training log structure: runs/EXP_NAME/checkpoints/model.pth -> runs/EXP_NAME/config.yaml).
   - Fallback: If not found, it defaults to 'configs/train_grpo.yaml'.
3. --nsample: Number of motion samples to generate per input text.
   - Use nsample > 1 to calculate best-of-n.
4. --batch_size: Batch size for evaluation.


python test.py --ckpt /home/allen/Diffusion/ReasonMotion_SFT_GRPO_Trajectory/runs/0130_lr_decay/checkpoints/checkpoint_ep1_batch12000.pth
"""

import argparse
import os
import torch
import yaml
from torch.utils.data import DataLoader
from model import ModelMain
from motion_data.finefs import FineFS

from utils.metrics import MetricsEvaluator

def load_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# 讀 config

def main():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--nsample", type=int, default=1, help="Number of samples generated per input")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    
    args = parser.parse_args()

    # Resolve config path if not specified
    if args.config is None:
        derived_config = os.path.join(os.path.dirname(args.ckpt), "..", "config.yaml")
        if os.path.exists(derived_config):
            print(f"found config file at {derived_config}")
            args.config = derived_config
        else:
            args.config = "configs/train_grpo.yaml"
            print(f"config file not found at {derived_config}, using default: {args.config}")

    # Check device
    device_name = args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        device_name = "cpu"
    device = torch.device(device_name)
    
    # Load Configuration
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    config = load_config(args.config)
    
    # Initialize Model
    print("🧠 Initializing Model...")
    target_dim = config['model'].get('target_dim', 72)
    model = ModelMain(config, device=device, target_dim=target_dim)
    model = model.to(device)
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    print(f"📥 Loading checkpoint: {args.ckpt}")
    state_dict = torch.load(args.ckpt, map_location=device)
    
    # Handle DataParallel prefix
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        
    try:
        model.load_state_dict(new_state_dict)
        print("✅ Checkpoint loaded successfully.")
    except Exception as e:
        print(f"⚠️ Strict loading failed, trying strict=False...")
        model.load_state_dict(new_state_dict, strict=False)
        print("✅ Checkpoint loaded (strict=False).")

    model.eval()
    
    # Load Validation Dataset (Split 2 for Test partition)
    print("📂 Loading Dataset (Split 2)...")
    data_cfg = config['data']
    data_name = data_cfg.get('name', 'FineFS').lower()
    
    if data_name == 'h36m':
        from motion_data.h36m import H36M
        val_dataset = H36M(
            data_dir=data_cfg['data_dir'],
            input_n=data_cfg['input_n'],
            output_n=data_cfg['output_n'],
            skip_rate=40,
            split=2,
            joints=data_cfg.get('joints', 17)
        )
    else:
        val_dataset = FineFS(
            data_dir=data_cfg['data_dir'],
            input_n=data_cfg['input_n'],
            output_n=data_cfg['output_n'],
            skip_rate=data_cfg.get('skip_rate', 5),
            mode=data_cfg['mode'],
            max_len=data_cfg.get('max_len', 90),
            split=2 
        )
    
    dataloader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4
    )
    
    evaluator = MetricsEvaluator(config=config, device=device)
    
    # Run Evaluation
    print(f"🚀 Starting Evaluation (n_sample={args.nsample})...")
    try:
        metrics = evaluator.evaluate(
            model=model,
            dataloader=dataloader,
            nsample=args.nsample
        )
        
        print("\n" + "="*40)
        print(f"{'METRIC':<20} | {'VALUE':<15}")
        print("-" * 40)
        for metric, value in metrics.items():
            print(f"{metric:<20} | {value:.6f}")
        print("="*40 + "\n")
        print(f"Results for: {args.ckpt}")
        
    except KeyboardInterrupt:
        print("\n❌ Evaluation interrupted.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n❌ Evaluation failed: {e}")

if __name__ == "__main__":
    main()