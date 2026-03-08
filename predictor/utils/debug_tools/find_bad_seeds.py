
import os
import torch
import numpy as np
import argparse
import sys
import yaml

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from model import ModelMain
from utils.rl_utils import UnifiedRewardModel
from utils.rl_visualizer import RLVisualizer
from utils.text_encoder import TextEncoder

def load_config(config_path):
    if not os.path.exists(config_path):
        # Try to resolve relative to project root
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        config_path = os.path.join(project_root, config_path)
        
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Find bad seeds for RL training initialization")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the target motion file (.pk)")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the base model SFT checkpoint")
    parser.add_argument("--reward_ckpt", type=str, required=True, help="Path to the FS reward model checkpoint")
    parser.add_argument("--config_path", type=str, default="configs/train_grpo.yaml", help="Path to config file for model params")
    parser.add_argument("--output_dir", type=str, default="analysis_bad_seeds", help="Output directory")
    parser.add_argument("--num_seeds", type=int, default=20, help="Number of random seeds to test")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    
    print(f"🔧 Loading Config from {args.config_path}...")
    config = load_config(args.config_path)
    
    # 1. Setup Dataset & Load Sample
    print(f"📂 Setup FineFS Dataset...")
    from utils.finefs import FineFS
    
    # We need to initialize FineFS with validation split
    # Config parameters are available in 'config'
    dataset = FineFS(
        data_dir=config['data']['data_dir'],
        input_n=config['data']['input_n'],
        output_n=config['data']['output_n'],
        skip_rate=config['data']['skip_rate'],
        mode=config['data']['mode'],
        max_len=config['data']['max_len'],
        split=1 # Validation
    )
    
    # Find sample index
    target_idx = 0
    found = False
    
    # Try to find the specific file in the dataset registry
    search_path = args.data_path
    print(f"🔍 Searching for sample: {search_path}")
    
    # FineFS structure check
    # Usually: dataset.data_idx is a list of keys
    # dataset.file_paths is a dict mapping key -> path
    if hasattr(dataset, 'file_paths'):
        for i, (key, _) in enumerate(dataset.data_idx):
            # Check if path contains our search string
            # Be robust: search path might be absolute, dataset path might be relative or absolute
            p = dataset.file_paths.get(key, "")
            if search_path in p or os.path.basename(search_path) in p:
                if search_path in p: # Strong match
                    target_idx = i
                    found = True
                    print(f"   ✅ Found exact match at index {i}: {p}")
                    break
                elif not found and os.path.basename(search_path) in p: # Weak match
                    target_idx = i
                    found = True # Keep searching for better match though? No break for now.
                    print(f"   ⚠️ Found weak match at index {i}: {p}")
                    break
    
    if not found:
        print(f"   ❌ Warning: Could not find target path in dataset. Using index 0.")
        target_idx = 0
        
    print(f"   📥 Loading sample index {target_idx}...")
    sample = dataset[target_idx]
    
    # Prepare Batch (Add Batch Dim)
    # Sample has keys: 'pose', 'timepoints', 'mask', 'motion_name'
    # pose: (T, K) numpy
    pose = torch.from_numpy(sample["pose"]).float().unsqueeze(0).to(device)
    tp = torch.from_numpy(sample["timepoints"]).float().unsqueeze(0).to(device)
    mask = torch.from_numpy(sample["mask"]).float().unsqueeze(0).to(device) # Original Mask (1 for obs, 0 for target)
    motion_name = sample["motion_name"]

    # For generation, we usually want to generate the unobserved part.
    # Model expects 'mask' to be 1 where observed (input).
    # So we can just use the sample mask directly.
    
    batch = {
        "pose": pose,
        "mask": mask, # Used for conditioning input
        "timepoints": tp,
        "motion_name": [motion_name]
    }
    
    # 2. Load Models
    print("🧠 Initializing Models...")
    # Update config strictly for loading
    config['pretrained_ckpt'] = args.ckpt_path
    config['rl']['fs_reward_ckpt'] = args.reward_ckpt
    config['rl']['w_score'] = 1.0 # Ensure Score model is active
    
    model = ModelMain(config, device=device, target_dim=72).to(device)
    
    # Load Weights
    print(f"📥 Loading Checkpoint: {args.ckpt_path}")
    state_dict = torch.load(args.ckpt_path, map_location=device)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    
    # Reward Model
    reward_model = UnifiedRewardModel(config=config, device=device)
    
    # Text Encoder
    text_encoder = TextEncoder(device=str(device))
    tok_emb, tok_mask = text_encoder(batch["motion_name"])
    text_cond = (tok_emb.to(device), tok_mask.to(device))
    
    # 3. Sampling Loop
    print(f"🎲 Running {args.num_seeds} seeds...")
    results = []
    
    # Prepare Feed (Common Inputs)
    feed = {"pose": batch["pose"], "mask": batch["mask"], "timepoints": batch["timepoints"]}
    gt_pose = batch["pose"] # (1, T, K)
    # Reshape GT for Reward Model: (1, 1, T, J, 3)
    B, T, K = gt_pose.shape
    gt_5d = gt_pose.reshape(B, 1, T, K//3, 3)
    
    for i in range(args.num_seeds):
        seed = 2026 + i
        torch.manual_seed(seed)
        
        with torch.no_grad():
            # Run Inference
            # output: (1, 1, K, L) -> (1, K, L) -> (1, L, K)
            # evaluate returns samples, pose, mask, tp
            out = model.evaluate(feed, 1, text_embedding=text_cond, sample=True)[0]
            
            sample_tk = out[0, 0].permute(1, 0).unsqueeze(0) # (1, T, K)
            
            # Calculate Score
            # format for reward: (B, G, T, J, 3)
            sample_5d = sample_tk.reshape(1, 1, T, K//3, 3)
            
            # Get Score Component Only
            r_score = reward_model.compute_score_reward(sample_5d).item()
            
            # Get Total (Weighted)
            r_total, metrics = reward_model(sample_5d, gt_5d)
            total_score = r_total.item()
            
            print(f"   Seed {seed}: Score={r_score:.4f}, Total={total_score:.4f}")
            
            results.append({
                "seed": seed,
                "score": r_score,
                "total": total_score,
                "data": sample_tk.cpu().numpy().reshape(T, K)
            })
            
    # 4. Sorting & Analysis
    # Sort by FS Score (ascending - worst first)
    results.sort(key=lambda x: x["score"])
    
    best = results[-1]
    worst = results[0]
    median = results[len(results)//2]
    
    print("\n📊 Summary:")
    print(f"   🏆 Best Seed {best['seed']} (Score: {best['score']:.4f})")
    print(f"   💩 Worst Seed {worst['seed']} (Score: {worst['score']:.4f})")
    print(f"   ⚖️ Median Seed {median['seed']} (Score: {median['score']:.4f})")
    
    # Save Log
    log_path = os.path.join(args.output_dir, "seed_analysis.txt")
    with open(log_path, 'w') as f:
        f.write("Seed,FS_Score,Total_Reward\n")
        for res in results:
            f.write(f"{res['seed']},{res['score']:.4f},{res['total']:.4f}\n")
    print(f"📝 Log saved to {log_path}")
    
    # 5. Visualization
    visualizer = RLVisualizer(output_dir=args.output_dir, device="cpu") # Use CPU for plotting
    
    # Construct Trajectories list for render_video
    trajectories = []
    
    # GT
    gt_data = gt_pose[0].cpu().numpy().reshape(-1, 24, 3)
    trajectories.append({
        "data": gt_data, "color": "blue", "alpha": 0.8, 
        "label": "GT", "linewidth": 2.0, "offset": np.array([0, 0, 0])
    })
    
    # Worst (Red/Orange)
    worst_data = worst["data"].reshape(-1, 24, 3)
    trajectories.append({
        "data": worst_data, "color": "red", "alpha": 0.9, 
        "label": f"Worst (Seed {worst['seed']}, S={worst['score']:.2f})", "linewidth": 2.0, "offset": np.array([0.5, 0, 0])
    })
    
    # Median (Yellow/Gold)
    median_data = median["data"].reshape(-1, 24, 3)
    trajectories.append({
        "data": median_data, "color": "gold", "alpha": 0.9, 
        "label": f"Median (Seed {median['seed']}, S={median['score']:.2f})", "linewidth": 2.0, "offset": np.array([1.0, 0, 0])
    })

    # Best (Green)
    best_data = best["data"].reshape(-1, 24, 3)
    trajectories.append({
        "data": best_data, "color": "green", "alpha": 0.9, 
        "label": f"Best (Seed {best['seed']}, S={best['score']:.2f})", "linewidth": 2.0, "offset": np.array([1.5, 0, 0])
    })
    
    visualizer.render_video("seed_comparison", trajectories, gt_data, mode_name="seed_analysis")

if __name__ == "__main__":
    main()
