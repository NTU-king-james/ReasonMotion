"""
Visualize model predictions across training batches AND text conditions in a 2D grid.

Grid Layout (viewed from 3/4 angle):
  - Columns (X axis): GT + batch checkpoints (training progression →)
  - Rows   (Depth):   different text conditions (single / double / triple / quadruple ↓)

Example (8 batches × 4 texts = 32 predictions + 4 GT references):
  ┌─────────────────────────────────────────────────────────────┐
  │            GT    B0     B300   B600   B900  ...  B2700      │
  │  single   🔵    🔴→→→→→→→→→→→→→→→→→→→→→→→→→→→→→🔴         │
  │  double   🔵    🟠→→→→→→→→→→→→→→→→→→→→→→→→→→→→→🟠         │
  │  triple   🔵    🟣→→→→→→→→→→→→→→→→→→→→→→→→→→→→→🟣         │
  │  quadrup  🔵    🟢→→→→→→→→→→→→→→→→→→→→→→→→→→→→→🟢         │
  └─────────────────────────────────────────────────────────────┘
  Color brightness & alpha increase with batch number (darker→brighter = earlier→later).

Usage:
  python visualize_batch_text_grid.py \\
    --run_dir "/home/allen/Diffusion/ReasonMotion_SFT_GRPO_Trajectory/runs/0212_balance_rewards" \\
    --res_pk  "/home/allen/datasets/FineFS_5s/3_final/valid/4F/4F_0011/new_res.pk" \\
    --batch_start 0 --batch_end 4500 --step 300 \\
    --seed 123 --slidewindow 10
"""

import os, pickle, argparse, random
import numpy as np
import torch, imageio, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from tqdm import tqdm

from model import ModelMain
from utils.text_encoder import TextEncoder
from utils.config_util import load_config
from utils.finefs import EDGES

# ============================== Constants ==============================
J = 24
DEFAULT_TEXTS = ["single", "double", "triple", "quadruple"]

# Distinct base colours (RGB) for each text condition
TEXT_BASE_COLORS = {
    "single":    np.array([1.0, 0.20, 0.20]),   # Red
    "double":    np.array([1.0, 0.55, 0.00]),   # Orange
    "triple":    np.array([0.55, 0.20, 1.00]),   # Purple
    "quadruple": np.array([0.10, 0.75, 0.20]),   # Green
}

# ============================== Helpers ==============================

def load_model_state(model, ckpt_path):
    print(f"  [Load] {ckpt_path}")
    sd = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(sd)


def load_model_for_batch(config, device, ckpt_path):
    model = ModelMain(config, device, target_dim=J * 3).to(device)
    load_model_state(model, ckpt_path)
    model.eval()
    return model


# ============================== Main ==============================

def render_grid_video(run_dir, res_pk, texts, batches, output_mp4, seed, slidewindow=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(os.path.join(run_dir, "config.yaml"))

    # ---------- reproducibility ----------
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    # ---------- load GT data ----------
    with open(res_pk, "rb") as f:
        data = pickle.load(f)
    key = "pred_xyz_24_struct_global" if "pred_xyz_24_struct_global" in data else "pred_xyz_24_struct"
    xyz = data[key].astype(np.float32)

    input_n  = config["data"]["input_n"]
    output_n = config["data"]["output_n"]
    total    = input_n + output_n
    xyz      = xyz[slidewindow : slidewindow + total]
    gt_pose  = xyz.reshape(-1, J * 3)  # (T, 72)

    # model feed (shared across all text conditions – only observed frames matter)
    gt_tensor   = torch.tensor(gt_pose).unsqueeze(0).to(device)        # (1,T,72)
    mask_tensor = torch.zeros_like(gt_tensor); mask_tensor[:, :input_n] = 1
    tp_tensor   = torch.arange(gt_tensor.shape[1]).unsqueeze(0).float().to(device)
    feed = {"pose": gt_tensor, "mask": mask_tensor, "timepoints": tp_tensor}

    # ---------- encode all text conditions ----------
    text_encoder = TextEncoder(device=device).to(device)
    text_conds = {}
    for txt in texts:
        with torch.no_grad():
            tok_emb, tok_mask = text_encoder([txt])
        text_conds[txt] = (tok_emb.to(device), tok_mask.to(device))

    # ==================== Collect predictions ====================
    predictions = {}  # (batch, text) -> ndarray (T, 72)
    print(f"🚀 Collecting {len(batches)} batches × {len(texts)} texts "
          f"= {len(batches) * len(texts)} predictions")

    for batch in tqdm(batches, desc="Batches"):
        ckpt_path = os.path.join(run_dir, "checkpoints",
                                 f"checkpoint_ep1_batch{batch}.pth")
        if not os.path.exists(ckpt_path):
            print(f"  ⚠️  not found: {ckpt_path}, skip")
            continue

        # load once per batch, run all text conditions
        model = load_model_for_batch(config, device, ckpt_path)

        for txt in texts:
            # reset seed before EACH generation → identical noise across batches
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

            with torch.no_grad():
                out = model.evaluate(feed, 1, text_embedding=text_conds[txt])[0]

            p = out[0, 0].cpu().numpy()   # (K, L)
            p = p.transpose(1, 0)          # (T, 72)
            predictions[(batch, txt)] = p

    if not predictions:
        print("❌ No predictions collected!"); return

    # ==================== Build trajectory list ====================
    trajectories = []
    sorted_batches = sorted({b for (b, _) in predictions.keys()})
    num_batches    = len(sorted_batches)

    x_spacing = 0.55   # column gap (data-X direction)
    z_spacing = 0.70   # row gap    (data-Z direction → visual depth)

    gt_data = gt_pose.reshape(-1, J, 3)

    # --- GT reference: one per text-row, at column 0 ---
    for row_i, txt in enumerate(texts):
        trajectories.append({
            "data":      gt_data,
            "color":     "royalblue",
            "alpha":     0.45,
            "linewidth": 1.5,
            "offset":    np.array([0.0, 0.0, row_i * z_spacing]),
        })

    # --- Predictions: col 1..N per text-row ---
    for row_i, txt in enumerate(texts):
        base_c = TEXT_BASE_COLORS.get(txt, np.array([0.5, 0.5, 0.5]))
        for col_i, batch in enumerate(sorted_batches):
            if (batch, txt) not in predictions:
                continue
            pred_data = predictions[(batch, txt)].reshape(-1, J, 3)

            ratio      = col_i / max(num_batches - 1, 1)
            brightness = 0.35 + 0.65 * ratio          # dark → bright
            color      = tuple(np.clip(base_c * brightness, 0, 1))
            alpha      = 0.45 + 0.55 * ratio          # transparent → opaque

            offset = np.array([(col_i + 1) * x_spacing, 0.0, row_i * z_spacing])
            trajectories.append({
                "data":      pred_data,
                "color":     color,
                "alpha":     alpha,
                "linewidth": 1.5,
                "offset":    offset,
            })

    # ==================== Render frames ====================
    print("🎥 Rendering grid video …")
    seq_len = gt_data.shape[0]

    # --- global bounds ---
    all_coords = np.concatenate([t["data"] + t["offset"] for t in trajectories], axis=0)
    min_v = np.min(all_coords, axis=(0, 1))
    max_v = np.max(all_coords, axis=(0, 1))

    pad = 0.5
    x_lim = [min(-pad, min_v[0]) - 0.1, max(pad, max_v[0]) + 0.1]
    z_lim = [min(-pad, min_v[2]) - 0.1, max(pad, max_v[2]) + 0.1]
    y_lim = [-0.5, 0.5]

    lx = x_lim[1] - x_lim[0]
    ly = z_lim[1] - z_lim[0]
    lz = y_lim[1] - y_lim[0]

    fig = plt.figure(figsize=(16, 9))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect((lx, ly, lz))

    # --- legend ---
    legend_handles = [mlines.Line2D([], [], color="royalblue", label="GT")]
    for txt in texts:
        c = tuple(TEXT_BASE_COLORS.get(txt, np.array([0.5, 0.5, 0.5])))
        legend_handles.append(mlines.Line2D([], [], color=c,
                                            label=txt.capitalize()))

    # pre-compute label positions (3D text annotations)
    # column headers: above the first text-row
    col_label_y = z_lim[0] - 0.05   # slightly in front (visual-Y = data-Z)
    col_label_z = y_lim[1]           # at top (visual-Z = -data-Y)
    # row headers: to the left of GT column
    row_label_x = x_lim[0] - 0.05

    print(f"Rendering {seq_len} frames → {output_mp4}")
    frames = []

    for t_idx in tqdm(range(seq_len), desc="Frames"):
        ax.clear()
        ax.set_xlim(x_lim)
        ax.set_ylim(z_lim)
        ax.set_zlim(y_lim)
        ax.set_xlabel("X"); ax.set_ylabel("Depth"); ax.set_zlabel("Height")
        ax.set_title(f"Batch × Text Grid | Frame {t_idx}/{seq_len}")
        ax.view_init(elev=20, azim=-60)
        ax.dist = 7.0

        # ---- draw skeletons ----
        for traj in trajectories:
            pose = traj["data"][t_idx] + traj["offset"]
            xs =  pose[:, 0]
            ys =  pose[:, 2]   # data-Z → vis-Y
            zs = -pose[:, 1]   # -data-Y → vis-Z

            ax.scatter(xs, ys, zs, c=[traj["color"]], s=12, alpha=traj["alpha"])
            for v1, v2 in EDGES:
                ax.plot([xs[v1], xs[v2]],
                        [ys[v1], ys[v2]],
                        [zs[v1], zs[v2]],
                        color=traj["color"],
                        alpha=traj["alpha"],
                        linewidth=traj["linewidth"])

        # ---- column labels (GT + batch numbers) ----
        ax.text(0, col_label_y, col_label_z, "GT",
                fontsize=7, ha="center", color="royalblue", fontweight="bold")
        for col_i, batch in enumerate(sorted_batches):
            ax.text((col_i + 1) * x_spacing, col_label_y, col_label_z,
                    f"B{batch}", fontsize=6, ha="center", color="gray")

        # ---- row labels (text conditions) ----
        for row_i, txt in enumerate(texts):
            c = tuple(TEXT_BASE_COLORS.get(txt, np.array([0.5, 0.5, 0.5])))
            ax.text(row_label_x, row_i * z_spacing, col_label_z,
                    txt.capitalize(), fontsize=7, ha="right", color=c,
                    fontweight="bold")

        ax.legend(handles=legend_handles, loc="upper right", fontsize=7)

        # ---- capture frame ----
        fig.canvas.draw()
        try:
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype="uint8")
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            frames.append(buf[:, :, :3].copy())
        except Exception as e:
            print(f"Frame capture failed: {e}"); break

    plt.close(fig)

    if frames:
        imageio.mimsave(output_mp4, frames, fps=30)
        print(f"✅ Saved to: {output_mp4}")
    else:
        print("❌ No frames generated.")


# ============================== CLI ==============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch × Text grid evolution visualizer")
    parser.add_argument("--run_dir", required=True,
                        help="Training run directory (with config.yaml & checkpoints/)")
    parser.add_argument("--res_pk", required=True,
                        help="Pickle file with GT skeleton data")
    parser.add_argument("--texts", nargs="+", default=DEFAULT_TEXTS,
                        help="Text conditions (default: single double triple quadruple)")
    parser.add_argument("--batch_start", type=int, default=0)
    parser.add_argument("--batch_end",   type=int, default=3000)
    parser.add_argument("--step",        type=int, default=300,
                        help="Batch interval between checkpoints")
    parser.add_argument("--seed",        type=int, default=123)
    parser.add_argument("--slidewindow", type=int, default=0)
    args = parser.parse_args()

    batches = list(range(args.batch_start, args.batch_end + 1, args.step))

    tag = f"grid_seed{args.seed}_w{args.slidewindow}"
    output_name = f"{tag}.mp4"
    output_path = os.path.join(args.run_dir, "visualize", output_name)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    render_grid_video(args.run_dir, args.res_pk, args.texts,
                      batches, output_path, args.seed, args.slidewindow)
