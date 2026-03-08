
import os, json, pickle, datetime, argparse, random
from pathlib import Path
import numpy as np
import torch, imageio, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import ModelMain
from utils.text_encoder import TextEncoder
from utils.config_util import load_config
from utils.finefs import EDGES

"""
python visualize_every_epoch.py \
  --run_dir "/home/allen/Diffusion/ReasonMotion_SFT_GRPO/runs/0102_SFT_RL_FineFS_vis_01" \
  --res_pk "/home/allen/datasets/FineFS_5s/3_final/valid/4F/4F_0011/new_res.pk" \
  --text "quadruple" \
  --epoch_start 1 \
  --epoch_end 50 \
  --step 5 \
  --seed 123 --slidewindow 0
"""

# ================ Skeleton & Draw ==================
J = 24

def load_model_state(model, ckpt_path):
    print(f"[Load model weights] {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)

def load_model_for_epoch(config, device, ckpt_path):
    model = ModelMain(config, device, target_dim=J * 3).to(device)
    load_model_state(model, ckpt_path)
    model.eval() # Use Eval mode to see the deterministic shift learned by RL
    return model

def safe_name(s:str)->str:
    return (s.replace(' ','_').replace('/','_')
             .replace('+','p').replace(',','')
             .replace('=','').replace('.','d'))



def render_evolution_video(run_dir, res_pk, text, epochs, output_mp4, seed, slidewindow=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_path = os.path.join(run_dir, "config.yaml")
    config = load_config(config_path)
    
    # Setup Randomness
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    
    # Load Data
    with open(res_pk, "rb") as f:
        data = pickle.load(f)
    key = 'pred_xyz_24_struct_global' if 'pred_xyz_24_struct_global' in data else 'pred_xyz_24_struct'
    xyz = data[key].astype(np.float32)
    
    input_n = config['data']['input_n']
    output_s = config['data']['output_n']
    total = input_n + output_s
    xyz = xyz[slidewindow:slidewindow+total]
    gt_pose = xyz.reshape(-1, J * 3) # (T, 72)
    
    # Pre-calculate GT feed
    # GT needs to be repeated for model sizing but we process one by one
    gt_tensor = torch.tensor(gt_pose).unsqueeze(0).to(device) # (1, T, 72)
    mask_tensor = torch.zeros_like(gt_tensor); mask_tensor[:, :input_n] = 1
    tp_tensor = torch.arange(gt_tensor.shape[1]).unsqueeze(0).float().to(device)
    feed = {"pose": gt_tensor, "mask": mask_tensor, "timepoints": tp_tensor}
    
    # Text Embedding
    text_encoder = TextEncoder(device=device).to(device)
    with torch.no_grad():
        tok_emb, tok_mask = text_encoder([text])
    text_cond = (tok_emb.to(device), tok_mask.to(device))
    
    # ================= Collect Predictions from Epochs =================
    predictions = {} # epoch -> (T, 72)
    
    print(f"🚀 Collecting generations for epochs: {epochs}")
    for ep in tqdm(epochs):
        ckpt_path = os.path.join(run_dir, "checkpoints", f"model_epoch_{ep}.pth")
        if not os.path.exists(ckpt_path):
            print(f"⚠️ Checkpoint not found: {ckpt_path}, skipping...")
            continue
            
        # Re-load model specifically for this epoch to ensure weights are fresh
        # (Could optimize by loading state dict into same model object, but safe is better)
        model = load_model_for_epoch(config, device, ckpt_path)
        
        # Ensure seed consistency FOR EACH GENERATION 
        # (Crucial: reset seed before each generation so noise is identical across epochs)
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        
        with torch.no_grad():
            out = model.evaluate(feed, 1, text_embedding=text_cond)[0] # (1, 1, K, T) or (1, 1, T, K) depending on verify
            # model.evaluate returns samples (B, N, K, T) ? No, see model.py evaluate:
            # samples = self.impute(...) -> (B, n, K, L)
            pass
        
        # Check Output Shape
        # impute returns (B, n, K, L) where L=num_frames
        # We want (T, K)
        p = out[0, 0].cpu().numpy() # (K, L)
        p = p.transpose(1, 0) # (L, K) -> (T, 72)
        predictions[ep] = p
        
    if not predictions:
        print("❌ No predictions collected!")
        return

    # ================= Render Video (RL Visualizer Logic) =================
    print("🎥 Rendering combined video using RL Visualizer Logic...")
    
    # 1. Trajectory Preparation
    trajectories = []
    
    # GT Trajectory (Blue)
    # GT Pose: (T, 72) -> (T, 24, 3)
    gt_data = gt_pose.reshape(-1, J, 3)
    trajectories.append({
        "data": gt_data,
        "color": "blue",
        "alpha": 1.0,
        "label": "GT",
        "linewidth": 2.0,
        "offset": np.array([0., 0., 0.])
    })
    
    sorted_epochs = sorted(predictions.keys())
    
    # Prediction Trajectories (Gradient Green)
    for i, ep in enumerate(sorted_epochs):
        offset = np.array([(i + 1) * 0.8, 0., 0.])
        pred_data = predictions[ep].reshape(-1, J, 3)
        
        # Gradient Logic
        ratio = i / max(len(sorted_epochs) - 1, 1)
        # R=0, G=0.5->1.0, B=0
        # Alpha=0.3->1.0
        c_val = 0.4 + 0.6 * ratio
        color = (0, c_val, 0) # Simple RGB tuple for matplotlib scatter/plot
        alpha = 0.5 + 0.5 * ratio
        
        trajectories.append({
            "data": pred_data,
            "color": color, 
            "alpha": alpha, 
            "label": f"Ep {ep}", 
            "linewidth": 1.5, 
            "offset": offset
        })

    # 2. Rendering Loop (Adapted from rl_visualizer.py)
    frames = []
    seq_len = gt_data.shape[0]
    
    # Calculate Global Bounds
    all_coords_list = []
    for t in trajectories:
        adjusted = t["data"] + t["offset"]
        all_coords_list.append(adjusted)
    all_coords = np.concatenate(all_coords_list, axis=0) # (Total, 24, 3)
    
    min_vals = np.min(all_coords, axis=(0, 1))
    max_vals = np.max(all_coords, axis=(0, 1))
    
    bound = 1.0
    # RL Visualizer mapping: X=x, Y=z, Z=-y
    # So bounds refer to Data X, Data Z, -Data Y
    x_lim = [min(-bound, min_vals[0]), max(bound, max_vals[0])]
    z_lim = [min(-bound, min_vals[2]), max(bound, max_vals[2])] # Data Z (Depth)
    y_lim = [min(-bound, -max_vals[1]), max(bound, -min_vals[1])] # -Data Y (Height)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    print(f"Rendering {seq_len} frames to {output_mp4}")
    
    for t in tqdm(range(seq_len), desc="Rendering frames"):
        ax.clear()
        
        ax.set_xlim(x_lim)
        ax.set_ylim(z_lim) # Visual Y = Data Z
        ax.set_zlim(y_lim) # Visual Z = -Data Y
        
        ax.set_xlabel('X')
        ax.set_ylabel('Z (Depth)')
        ax.set_zlabel('Y (Height)')
        ax.set_title(f"Evolution | Frame {t}/{seq_len}")
        
        ax.view_init(elev=10, azim=-90)
        
        # Legend elements
        import matplotlib.lines as mlines
        legend_handles = []
        # Add GT
        legend_handles.append(mlines.Line2D([], [], color='blue', label='GT'))
        # Add First and Last Epoch for brevity
        if sorted_epochs:
            legend_handles.append(mlines.Line2D([], [], color='green', label=f'Ep {sorted_epochs[0]} - {sorted_epochs[-1]}'))

        for traj in trajectories:
            pose = traj["data"][t] # (24, 3)
            pose_off = pose + traj["offset"]
            
            xs = pose_off[:, 0]
            ys = pose_off[:, 2]  # Data Z -> Vis Y
            zs = -pose_off[:, 1] # -Data Y -> Vis Z
            
            ax.scatter(xs, ys, zs, c=traj["color"], s=15, alpha=traj["alpha"])
            
            for (v1, v2) in EDGES:
                 x_pair = [xs[v1], xs[v2]]
                 y_pair = [ys[v1], ys[v2]]
                 z_pair = [zs[v1], zs[v2]]
                 ax.plot(x_pair, y_pair, z_pair, color=traj["color"], 
                         alpha=traj["alpha"], linewidth=traj["linewidth"])

        ax.legend(handles=legend_handles, loc='upper right')
        
        fig.canvas.draw()
        try:
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(image)
        except Exception as e:
            print(f"Frame capture failed: {e}")
            break
            
    plt.close(fig)
    
    if frames:
        imageio.mimsave(output_mp4, frames, fps=30)
        print(f"✅ Saved to: {output_mp4}")
    else:
        print("❌ Video generation failed (no frames).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--res_pk", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--epoch_start", type=int, default=1)
    parser.add_argument("--epoch_end", type=int, default=50)
    parser.add_argument("--step", type=int, default=5, help="Epoch interval (e.g. every 5 epochs)")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--slidewindow", type=int, default=0)
    args = parser.parse_args()
    
    epochs = list(range(args.epoch_start, args.epoch_end, args.step))
    output_name = f"evolution_every_epoch_seed{args.seed}.mp4"
    output_path = os.path.join(args.run_dir, "visualize", output_name)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    render_evolution_video(args.run_dir, args.res_pk, args.text, epochs, output_path, args.seed, args.slidewindow)
