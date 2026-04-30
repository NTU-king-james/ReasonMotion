
import os, json, pickle, datetime, argparse, random
import sys
from pathlib import Path
import numpy as np
import torch, imageio, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# Ensure imports work whether the script is launched from predictor/ or predictor/visualize/.
SCRIPT_DIR = Path(__file__).resolve().parent
PREDICTOR_ROOT = SCRIPT_DIR.parent
if str(PREDICTOR_ROOT) not in sys.path:
    sys.path.insert(0, str(PREDICTOR_ROOT))

from model.model import ModelMain
from motion_data.h36m_unified import H36MUnified
from utils.text_encoder import TextEncoder
from utils.config_util import load_config

FINEFS_EDGES = [
    (0, 1), (1, 4), (4, 7), (7, 10), (0, 2), (2, 5), (5, 8), (8, 11),
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15), (12, 13), (13, 16), (16, 18),
    (18, 20), (20, 22), (12, 14), (14, 17), (17, 19), (19, 21), (21, 23)
]

H36M_EDGES = [
    (0, 1), (1, 2), (2, 3),        # r-leg
    (0, 4), (4, 5), (5, 6),        # l-leg
    (0, 7), (7, 8), (8, 9), (9, 10), # spine/head
    (8, 11), (11, 12), (12, 13),   # l-arm
    (8, 14), (14, 15), (15, 16)     # r-arm
]

"""
python visualize_every_batch.py \
  --run_dir "/home/allen/Diffusion/ReasonMotion_SFT_GRPO_Trajectory/runs/0128_balance_reward_from_bad_seed" \
  --res_pk "/home/allen/datasets/FineFS_5s/3_final/valid/4F/4F_0011/new_res.pk" \
  --text "quadruple" \
  --batch_start 0 \
  --batch_end 3000 \
  --step 300 \
  --seed 123 --slidewindow 10

python visualize_every_batch.py \
    --mode h36m \
    --run_dir "/home/kingjames23/ReasonMotion/predictor/runs/h36m_fairscale_rl_with_smooth" \
    --data_dir "/home/allen/datasets" \
    --sample_idx 0 \
    --batch_start 0 \
    --batch_end 1000 \
    --step 200 \
    --seed 123
"""

def infer_joint_count(xyz: np.ndarray, config: dict) -> int:
    """Infer the joint count from motion data or config."""
    if xyz.ndim == 3:
        return int(xyz.shape[1])
    if xyz.ndim == 2 and xyz.shape[1] % 3 == 0:
        return int(xyz.shape[1] // 3)

    target_dim = int(config.get("model", {}).get("target_dim", 72))
    if target_dim % 3 == 0:
        return target_dim // 3
    raise ValueError(f"Unable to infer joint count from data shape {xyz.shape}")


def get_edges(joints: int):
    if joints == 24:
        return FINEFS_EDGES
    if joints == 17:
        return H36M_EDGES
    return []

def load_model_state(model, ckpt_path):
    print(f"[Load model weights] {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)

def load_model_for_epoch(config, device, ckpt_path, target_dim):
    model = ModelMain(config, device, target_dim=target_dim).to(device)
    load_model_state(model, ckpt_path)
    model.eval() # Use Eval mode to see the deterministic shift learned by RL
    return model


def compute_reward_components(pred_pose: torch.Tensor, gt_pose: torch.Tensor):
    """Match the RL reward definitions for GT alignment and smoothness."""
    diff = pred_pose - gt_pose
    dist = torch.norm(diff, dim=-1).mean(dim=(-1, -2))
    r_gt = torch.exp(-1.0 * dist)

    vel = torch.diff(pred_pose, dim=2)
    acc = torch.diff(vel, dim=2)
    acc_mag = torch.norm(acc, dim=-1).mean(dim=(-1, -2))
    r_smooth = torch.exp(-10.0 * acc_mag)

    return r_gt, r_smooth

def safe_name(s:str)->str:
    return (s.replace(' ','_').replace('/','_')
             .replace('+','p').replace(',','')
             .replace('=','').replace('.','d'))


def build_finefs_reference(res_pk, slidewindow, total_frames):
    with open(res_pk, "rb") as f:
        data = pickle.load(f)

    key = 'pred_xyz_24_struct_global' if 'pred_xyz_24_struct_global' in data else 'pred_xyz_24_struct'
    if key not in data:
        tensor_like = [k for k, v in data.items() if hasattr(v, "shape")]
        if not tensor_like:
            raise KeyError("No motion array found in res_pk")
        key = tensor_like[0]

    xyz = np.asarray(data[key], dtype=np.float32)
    xyz = xyz[slidewindow:slidewindow + total_frames]
    joints = infer_joint_count(xyz, {})
    pose = xyz.reshape(-1, joints * 3)
    return pose, joints, None


def build_h36m_reference(config, data_dir, sample_idx, split, input_n, output_s):
    dataset = H36MUnified(
        data_dir=data_dir,
        input_n=input_n,
        output_n=output_s,
        skip_rate=config['data'].get('skip_rate', 1),
        split=split,
        joints=config['data'].get('joints', 17),
        downsample=config['data'].get('downsample', 1),
        max_len=config['data'].get('max_len'),
        no_overlap=config['data'].get('no_overlap', False),
        protocol=config['data'].get('h36m_protocol', 'predictor'),
        miss_type=config['data'].get('miss_type', 'no_miss'),
        miss_rate=config['data'].get('miss_rate', 0.2),
        all_data=config['data'].get('all_data', True),
        data_ratio=config['data'].get('data_ratio', 1.0),
        pad_short_sequences=config['data'].get('pad_short_sequences', False),
    )

    if sample_idx < 0 or sample_idx >= len(dataset):
        raise IndexError(f"sample_idx {sample_idx} out of range for H36M dataset length {len(dataset)}")

    sample = dataset[sample_idx]
    pose = np.asarray(sample["pose"], dtype=np.float32)
    if pose.ndim != 2:
        raise ValueError(f"Unexpected H36M pose shape: {pose.shape}")

    joints = pose.shape[1] // 3
    return pose, joints, sample



def render_evolution_video(run_dir, mode, res_pk, data_dir, text, batches, output_mp4, seed,
                           slidewindow=0, sample_idx=0, h36m_split=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_path = os.path.join(run_dir, "config.yaml")
    config = load_config(config_path)
    
    # Setup Randomness
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    if mode == "finefs":
        if not res_pk:
            raise ValueError("FineFS mode requires --res_pk")
        reference_pose, joints, sample = build_finefs_reference(
            res_pk=res_pk,
            slidewindow=slidewindow,
            total_frames=config['data']['input_n'] + config['data']['output_n'],
        )
        sample_text = text
        if sample_text is None:
            raise ValueError("FineFS mode requires --text")
    elif mode == "h36m":
        if not data_dir:
            raise ValueError("H36M mode requires --data_dir")
        reference_pose, joints, sample = build_h36m_reference(
            config=config,
            data_dir=data_dir,
            sample_idx=sample_idx,
            split=h36m_split,
            input_n=config['data']['input_n'],
            output_s=config['data']['output_n'],
        )
        sample_text = text or sample.get("motion_name", "unknown")
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    target_dim = joints * 3
    edges = get_edges(joints)
    
    input_n = config['data']['input_n']
    output_s = config['data']['output_n']
    total = input_n + output_s
    gt_pose = reference_pose[:total].reshape(-1, target_dim)
    
    # Pre-calculate GT feed
    # GT needs to be repeated for model sizing but we process one by one
    gt_tensor = torch.tensor(gt_pose).unsqueeze(0).to(device)
    mask_tensor = torch.zeros_like(gt_tensor); mask_tensor[:, :input_n] = 1
    tp_tensor = torch.arange(gt_tensor.shape[1]).unsqueeze(0).float().to(device)
    feed = {"pose": gt_tensor, "mask": mask_tensor, "timepoints": tp_tensor}
    
    # Text Embedding
    text_encoder = TextEncoder(device=device).to(device)
    with torch.no_grad():
        tok_emb, tok_mask = text_encoder([sample_text])
    text_cond = (tok_emb.to(device), tok_mask.to(device))
    
    # ================= Collect Predictions from Pretrained Base =================
    pretrained_ckpt = config.get("pretrained_ckpt", None)
    pretrained_pred = None
    if pretrained_ckpt and os.path.exists(pretrained_ckpt):
        print(f"🚀 Collecting generation for pretrained checkpoint: {pretrained_ckpt}")
        model = load_model_for_epoch(config, device, pretrained_ckpt, target_dim)
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        with torch.no_grad():
            out = model.evaluate(feed, 1, text_embedding=text_cond)[0]
        p = out[0, 0].cpu().numpy()
        p = p.transpose(1, 0)
        pretrained_pred = p
    elif pretrained_ckpt:
        print(f"⚠️ Pretrained checkpoint not found: {pretrained_ckpt}")

    # ================= Collect Predictions from Batches =================
    predictions = {} # batch -> (T, K)
    batch_metrics = {}
    
    print(f"🚀 Collecting generations for batches: {batches}")
    progress = tqdm(batches, desc=f"Collecting ({mode})")
    for batch in progress:
        ckpt_path = os.path.join(run_dir, "checkpoints",  f"checkpoint_ep1_batch{batch}.pth")
        if not os.path.exists(ckpt_path):
            print(f"⚠️ Checkpoint not found: {ckpt_path}, skipping...")
            progress.set_postfix({"R_GT": "skip", "R_Smooth": "skip"})
            continue
            
        # Re-load model specifically for this epoch to ensure weights are fresh
        # (Could optimize by loading state dict into same model object, but safe is better)
        model = load_model_for_epoch(config, device, ckpt_path, target_dim)
        
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
        p = out[0, 0].transpose(0, 1) # (T, K)
        p_np = p.cpu().numpy()
        predictions[batch] = p_np

        gt_t = gt_tensor[:, :p.shape[0], :].unsqueeze(1).reshape(1, 1, p.shape[0], joints, 3)
        pred_t = p.reshape(1, 1, p.shape[0], joints, 3)
        r_gt, r_smooth = compute_reward_components(pred_t, gt_t)
        r_gt_val = float(r_gt.item())
        r_smooth_val = float(r_smooth.item())
        batch_metrics[batch] = {"R_GT": r_gt_val, "R_Smooth": r_smooth_val}
        progress.set_postfix({
            "R_GT": f"{r_gt_val:.4f}",
            "R_Smooth": f"{r_smooth_val:.4f}",
        })
        
    if not predictions:
        print("❌ No predictions collected!")
        return

    if batch_metrics:
        avg_r_gt = sum(v["R_GT"] for v in batch_metrics.values()) / len(batch_metrics)
        avg_r_smooth = sum(v["R_Smooth"] for v in batch_metrics.values()) / len(batch_metrics)
        print(f"📊 Average over collected batches: R_GT={avg_r_gt:.4f}, R_Smooth={avg_r_smooth:.4f}")

    # ================= Render Video (RL Visualizer Logic) =================
    print("🎥 Rendering combined video using RL Visualizer Logic...")
    
    # 1. Trajectory Preparation
    trajectories = []
    
    # GT Trajectory (Blue)
    # GT Pose: (T, K) -> (T, J, 3)
    gt_data = gt_pose.reshape(-1, joints, 3)
    trajectories.append({
        "data": gt_data,
        "color": "blue",
        "alpha": 1.0,
        "label": "GT",
        "linewidth": 2.0,
        "offset": np.array([0., 0., 0.])
    })
    
    # Pretrained Trajectory (Red)
    if pretrained_pred is not None:
        pretrained_data = pretrained_pred.reshape(-1, joints, 3)
        trajectories.append({
            "data": pretrained_data,
            "color": "red",
            "alpha": 1.0,
            "label": "Base",
            "linewidth": 2.0,
            "offset": np.array([0.55, 0., 0.])
        })

    sorted_epochs = sorted(predictions.keys())
    
    # Prediction Trajectories (Gradient Green)
    for i, ep in enumerate(sorted_epochs):
        offset_idx = (i + 2) if pretrained_pred is not None else (i + 1)
        offset = np.array([offset_idx * 0.55, 0., 0.])
        pred_data = predictions[ep].reshape(-1, joints, 3)
        
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
            "label": f"Batch {ep}", 
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
    
    bound = 0.5
    # RL Visualizer mapping: X=x, Y=z, Z=-y
    # So bounds refer to Data X, Data Z, -Data Y
    x_lim = [min(-bound, min_vals[0]), max(bound, max_vals[0])]
    z_lim = [min(-bound, min_vals[2]), max(bound, max_vals[2])] # Data Z (Depth)
    y_lim = [-0.5, 0.5] # -Data Y (Height)

    lx = x_lim[1] - x_lim[0]
    ly = z_lim[1] - z_lim[0]
    lz = y_lim[1] - y_lim[0]

    # Revert to a fixed size to avoid getting too wide
    fig = plt.figure(figsize=(10, 6))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = fig.add_subplot(111, projection='3d')
    # Use equal aspect ratio to keep proportions correct but rely on camera view to frame it
    ax.set_box_aspect((lx, ly, lz))
    
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
        ax.dist = 7.0
        
        # Legend elements
        import matplotlib.lines as mlines
        legend_handles = []
        # Add GT
        legend_handles.append(mlines.Line2D([], [], color='blue', label='GT'))
        if pretrained_pred is not None:
            legend_handles.append(mlines.Line2D([], [], color='red', label='Base'))
        # Add First and Last Batch for brevity
        if sorted_epochs:
            legend_handles.append(mlines.Line2D([], [], color='green', label=f'Batch {sorted_epochs[0]} - {sorted_epochs[-1]}'))

        for traj in trajectories:
            pose = traj["data"][t] # (J, 3)
            pose_off = pose + traj["offset"]
            
            xs = pose_off[:, 0]
            ys = pose_off[:, 2]  # Data Z -> Vis Y
            zs = -pose_off[:, 1] # -Data Y -> Vis Z
            
            ax.scatter(xs, ys, zs, c=traj["color"], s=15, alpha=traj["alpha"])
            
            for (v1, v2) in edges:
                x_pair = [xs[v1], xs[v2]]
                y_pair = [ys[v1], ys[v2]]
                z_pair = [zs[v1], zs[v2]]
                ax.plot(x_pair, y_pair, z_pair, color=traj["color"], 
                        alpha=traj["alpha"], linewidth=traj["linewidth"])

        ax.legend(handles=legend_handles, loc='upper right')
        
        fig.canvas.draw()
        try:
            image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            image = image[:, :, :3].copy()
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
    parser.add_argument("--mode", type=str, default="finefs", choices=["finefs", "h36m"], help="Data mode")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--res_pk", default=None, help="FineFS mode only")
    parser.add_argument("--data_dir", default=None, help="H36M mode only")
    parser.add_argument("--text", default=None, help="FineFS prompt or optional H36M override text")
    parser.add_argument("--batch_start", type=int, default=1)
    parser.add_argument("--batch_end", type=int, default=50)
    parser.add_argument("--step", type=int, default=5, help="batch interval (e.g. every 5 epochs)")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--slidewindow", type=int, default=0)
    parser.add_argument("--sample_idx", type=int, default=0, help="H36M mode: fixed dataset index for comparison")
    parser.add_argument("--h36m_split", type=int, default=2, choices=[0, 1, 2], help="H36M split to use")
    args = parser.parse_args()
    
    batches = list(range(args.batch_start, args.batch_end+1, args.step))
    output_name = f"every_batch_seed{args.seed}_window{args.slidewindow}.mp4"
    output_path = os.path.join(args.run_dir, "visualize", output_name)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    render_evolution_video(
        run_dir=args.run_dir,
        mode=args.mode,
        res_pk=args.res_pk,
        data_dir=args.data_dir,
        text=args.text,
        batches=batches,
        output_mp4=output_path,
        seed=args.seed,
        slidewindow=args.slidewindow,
        sample_idx=args.sample_idx,
        h36m_split=args.h36m_split,
    )
