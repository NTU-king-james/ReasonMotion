
import os
import argparse
import random
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm
from pathlib import Path

# Add project root to path to ensure imports work
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from utils.finefs import FineFS
from utils.fs_reward_utils.fs_reward_model import FSRewardModel

"""
example usage:
python test_model/evaluate_reward.py \
  --test_all \
  --checkpoint /home/allen/Diffusion/ReasonMotion_SFT_GRPO/utils/fs_reward_utils/checkpoints_goe/best_model_goe.pth \
  --vis_count 5 \
  --output_dir test_model/results/full_eval \
  --layout SMPL_24
"""




# Skeleton definitions (from visualize_infer.py)
J = 24
EDGES = [
    (0,1),(1,4),(4,7),(7,10),
    (0,2),(2,5),(5,8),(8,11),
    (0,3),(3,6),(6,9),(9,12),
    (12,15),(12,13),(13,16),(16,18),(18,20),(20,22),
    (12,14),(14,17),(17,19),(19,21),(21,23)
]

def draw(ax, gt, bones, idx, text, score_text):
    """
    Draw skeleton on matplotlib 3d axis
    """
    ax.clear()
    floor = 0.8
    # Draw floor
    ax.plot_surface(*np.meshgrid(np.linspace(-floor,floor,2),
                                 np.linspace(-floor,floor,2)),
                    np.full((2,2),-0.6),color='lightgray',alpha=.15)
    
    # Draw skeleton
    # gt shape: (24, 3)
    # visualize_infer.py uses: (x, z, -y) or something similar based on view_init
    # In visualize_infer.py: 
    # ax.scatter(g[:,0],g[:,2],-g[:,1], ...)
    
    g = gt.copy()
    
    # Scatter joints
    ax.scatter(g[:,0], g[:,2], -g[:,1], c='blue', s=20)
    
    # Plot bones
    for a,b in bones: 
        ax.plot([g[a,0],g[b,0]], [g[a,2],g[b,2]], [-g[a,1],-g[b,1]], c='blue')
        
    ax.set_title(f"Frame {idx}\n{text}", fontsize=10)
    
    # Add score text in top right
    # transform=ax.transAxes makes coordinates relative to axes (0,0 is bottom-left, 1,1 is top-right)
    ax.text2D(0.95, 0.95, score_text, transform=ax.transAxes, 
              horizontalalignment='right', verticalalignment='top', 
              fontsize=12, color='red', bbox=dict(facecolor='white', alpha=0.7))

    ax.view_init(20, 60) # Adjusted view angle
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-0.6, 0.6])
    ax.set_zlim([-0.6, 0.6])
    ax.set_axis_off()

def save_video(pose, motion_name, pred_score, gt_score, save_path, fps=25):
    """
    Generate and save video for a single sample
    pose: (T, 24, 3) numpy array
    """
    T = pose.shape[0]
    writer = imageio.get_writer(str(save_path), fps=fps, codec='libx264', bitrate='12M')
    
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    
    score_str = f"Pred: {pred_score:.2f}\nGT: {gt_score:.2f}\nError: {abs(pred_score-gt_score):.2f}"
    
    print(f"Generating video for {motion_name}...")
    for t in range(T):
        draw(ax, pose[t], EDGES, t, motion_name, score_str)
        
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        img = img.reshape(h, w, 4)
        frame = img[:, :, :3] # RGBA -> RGB
        writer.append_data(frame)
    
    writer.close()
    plt.close(fig)
    print(f"Saved: {save_path}")

def generate_report(results, output_dir):
    """
    Generate a detailed text report of the evaluation results.
    results: list of dicts with keys: 'name', 'gt', 'pred', 'error'
    """
    if not results:
        print("No results to report.")
        return

    errors = [r['error'] for r in results]
    mse_vals = [r['error']**2 for r in results]
    
    mae = np.mean(errors)
    mse = np.mean(mse_vals)
    
    # Sort by error (descending)
    results_sorted = sorted(results, key=lambda x: x['error'], reverse=True)
    
    # Categorize
    overestimated = [r for r in results if r['pred'] > r['gt']]
    underestimated = [r for r in results if r['pred'] < r['gt']]
    
    # Sort categories by error magnitude
    overestimated.sort(key=lambda x: x['error'], reverse=True)
    underestimated.sort(key=lambda x: x['error'], reverse=True)
    
    report_path = os.path.join(output_dir, "evaluation_report.txt")
    
    with open(report_path, "w") as f:
        f.write("Evaluation Report\n")
        f.write("=================\n")
        f.write(f"Total Samples: {len(results)}\n")
        f.write(f"Mean Absolute Error (MAE): {mae:.4f}\n")
        f.write(f"Mean Squared Error (MSE): {mse:.4f}\n\n")
        
        f.write("Top 10 Overestimated (Pred > GT):\n")
        for i, r in enumerate(overestimated[:10]):
            f.write(f"  {i+1}. {r['name']} | GT: {r['gt']:.2f}, Pred: {r['pred']:.2f}, Diff: +{r['error']:.2f}\n")
        
        f.write("\nTop 10 Underestimated (Pred < GT):\n")
        for i, r in enumerate(underestimated[:10]):
            f.write(f"  {i+1}. {r['name']} | GT: {r['gt']:.2f}, Pred: {r['pred']:.2f}, Diff: -{r['error']:.2f}\n")
            
        f.write("\n\nFull Results (Top 50 Errors):\n")
        for i, r in enumerate(results_sorted[:50]):
             diff_sign = "+" if r['pred'] > r['gt'] else "-"
             f.write(f"  {i+1}. {r['name']} | GT: {r['gt']:.2f}, Pred: {r['pred']:.2f}, Error: {r['error']:.2f} ({diff_sign})\n")

    print(f"Report saved to: {report_path}")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to evaluate (ignored if --test_all is set)")
    parser.add_argument("--test_all", action="store_true", help="Evaluate all samples in the dataset split")
    parser.add_argument("--vis_count", type=int, default=10, help="Maximum number of videos to generate")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, default="/home/allen/datasets/FineFS_5s/3_final", help="Dataset root directory")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--split", type=int, default=2, help="Dataset split (2 for test)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--layout", type=str, default="SMPL", choices=["SMPL", "SMPL_24"], help="Skeleton layout (SMPL=22 joints, SMPL_24=24 joints)")
    
    args = parser.parse_args()
    
    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Load Dataset
    print(f"Loading dataset from {args.data_dir}...")
    dataset = FineFS(
        data_dir=args.data_dir,
        input_n=30,
        output_n=40,
        split=args.split,
        mode="rotation",
        move_global=True,
        disable_sliding=True,  # Testing: evaluate only one window per sequence
        random_face=False,     # Deterministic for evaluation
        reward_mode=True,      # Return (6,T,V) tensor + judge_score
    )
    print(f"Dataset loaded. Total samples: {len(dataset)}")
    
    # Select Samples
    if args.test_all:
        print("Evaluating ALL samples...")
        indices = list(range(len(dataset)))
    else:
        if args.num_samples > len(dataset):
            indices = list(range(len(dataset)))
        else:
            indices = random.sample(range(len(dataset)), args.num_samples)
        print(f"Evaluating {len(indices)} random samples...")

    # Shuffle indices to ensure random visualization order
    random.shuffle(indices)
    
    # Load Model
    print(f"Loading model from {args.checkpoint} with layout {args.layout}...")
    reward_model = FSRewardModel(
        checkpoint_path=args.checkpoint,
        device=str(device),
        scale_type='linear',
        min_score=0.0,
        max_score=20.0,
        layout=args.layout
    )
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    results = []
    vis_counter = 0
    
    print("\nStarting evaluation...")
    
    for idx_i, idx in enumerate(tqdm(indices, desc="Evaluating")):
        sample = dataset[idx]
        
        # Prepare input
        pose_in = sample['pose'].unsqueeze(0).to(device) # (1, 6, T, 24)
        
        # If model is 22 joints (SMPL), we need to slice
        if args.layout == 'SMPL':
             if pose_in.shape[-1] == 24:
                 pose_in_model = pose_in[:, :, :, :22]
             else:
                 pose_in_model = pose_in
        else:
             # SMPL_24 or other
             pose_in_model = pose_in

        gt_score = sample['judge_score'].item() # float
        motion_name = sample['motion_name']
        
        # Inference
        with torch.no_grad():
            pred_score = reward_model.scoring_model(pose_in_model).item()
            
        err = abs(pred_score - gt_score)
        
        results.append({
            'name': motion_name,
            'gt': gt_score,
            'pred': pred_score,
            'error': err,
            'idx': idx
        })
        
        # Random Visualization (limited by vis_count)
        if vis_counter < args.vis_count:
            pose_vis = sample['pose'][:3, :, :].permute(1, 2, 0).cpu().numpy() # (T, 24, 3)
            
            safe_name = motion_name.replace(" ", "_").replace("/", "-").replace("+", "p")
            vid_name = f"random_{vis_counter}_{safe_name}_err{err:.2f}.mp4"
            save_path = os.path.join(args.output_dir, vid_name)
            
            save_video(pose_vis, motion_name, pred_score, gt_score, save_path)
            vis_counter += 1
        
    generate_report(results, args.output_dir)

    # --- Visualize Top Errors ---
    print("\nGenerating videos for top errors...")
    
    # Sort categories
    overestimated = [r for r in results if r['pred'] > r['gt']]
    underestimated = [r for r in results if r['pred'] < r['gt']]
    
    overestimated.sort(key=lambda x: x['error'], reverse=True)
    underestimated.sort(key=lambda x: x['error'], reverse=True)
    
    top_n = 5 # Default top N to visualize
    
    # Function helper
    def save_top_videos(samples, prefix):
        for i, r in enumerate(samples[:top_n]):
            idx = r['idx']
            sample = dataset[idx] # Fetch data again
            pose_vis = sample['pose'][:3, :, :].permute(1, 2, 0).cpu().numpy()
            
            safe_name = r['name'].replace(" ", "_").replace("/", "-").replace("+", "p")
            vid_name = f"{prefix}_{i+1}_{safe_name}_err{r['error']:.2f}.mp4"
            save_path = os.path.join(args.output_dir, vid_name)
            
            save_video(pose_vis, r['name'], r['pred'], r['gt'], save_path)

    save_top_videos(overestimated, "top_over")
    save_top_videos(underestimated, "top_under")
    
    # --- Visualize Highest/Lowest Scores ---
    print("\nGenerating videos for highest/lowest scores...")
    
    # Sort by Ground Truth Score
    results_by_score = sorted(results, key=lambda x: x['gt'], reverse=True)
    
    highest_score_samples = results_by_score[:top_n]
    lowest_score_samples = results_by_score[-top_n:]
    
    save_top_videos(highest_score_samples, "highest_score")
    save_top_videos(lowest_score_samples, "lowest_score")
    
    print("Done.")

if __name__ == "__main__":
    main()
