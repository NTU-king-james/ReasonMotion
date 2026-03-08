"""
verify_finefs.py  —  FineFS Dataset 視覺化驗證工具

生成三支驗證影片，讓你肉眼確認 finefs.py 各功能是否正確：

  video1_random_face.mp4
      同一個 sliding window，正上方俯視圖，顯示 6 次不同的 random_face 結果。
      ✅ 預期：骨架形狀/品質相同，但水平朝向各異 (Y 軸旋轉)
      ❌ 如有問題：所有版本朝向相同 → augment 未生效

  video2_move_global.mp4
      左欄：move_global=True （全域座標，滑冰路徑 drift 可見）
      右欄：move_global=False（相機空間，根節點固定在原點附近）
      ✅ 預期：左欄可看到水平位移；右欄幾乎固定在原點
      ❌ 如有問題：兩欄完全相同 → 開關無效

  video3_mode_consistency.mp4
      同一個序列，左欄用 reward_mode=True 取出 joint channel (6,T,V)→前三通道，
      右欄用 reward_mode=False 取出 (T,V*3) reshape 成 (T,V,3)。
      ✅ 預期：兩邊骨架完全重疊 / 幾何一致
      ❌ 如有問題：骨架位置不同 → 兩種 mode 的座標系不一致

使用方式 (從專案根目錄執行):
  python utils/debug_tools/verify_finefs.py
  python utils/debug_tools/verify_finefs.py --data_dir /your/data --out_dir /your/output
"""
import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio
from pathlib import Path

PRJ_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PRJ_ROOT))
from utils.finefs import FineFS, EDGES

# ------------------------------------------------------------------ #
#  骨骼繪製
# ------------------------------------------------------------------ #
# Colour palette for random_face panels
PANEL_COLORS = ["#e63946", "#457b9d", "#2a9d8f", "#e9c46a", "#f4a261", "#6a4c93"]

def draw_skeleton_3d(ax, joints, title="", color="steelblue", elev=20, azim=60):
    """
    joints: (V, 3)  world coords (X right, Y up, Z forward)
    Visualise as: X=right, Z=depth, -Y=height
    """
    ax.clear()
    g = joints
    # Use plot+marker instead of scatter to avoid matplotlib 3D scatter API differences
    for v in range(g.shape[0]):
        ax.plot([g[v, 0]], [g[v, 2]], [-g[v, 1]],
                marker='o', color=color, markersize=3, linestyle='None')
    for a, b in EDGES:
        ax.plot([g[a, 0], g[b, 0]],
                [g[a, 2], g[b, 2]],
                [-g[a, 1], -g[b, 1]],
                c=color, lw=1.5)
    ax.set_title(title, fontsize=8, pad=2)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.0, 0.5])
    ax.set_axis_off()


def draw_skeleton_topdown(ax, joints, title="", color="steelblue"):
    """
    Overhead (top-down) view: X-Z plane, ignoring Y.
    Great for seeing horizontal rotation direction.
    joints: (V, 3)
    """
    ax.clear()
    g = joints
    ax.scatter(g[:, 0], g[:, 2], c=color, s=18)
    for a, b in EDGES:
        ax.plot([g[a, 0], g[b, 0]], [g[a, 2], g[b, 2]], c=color, lw=1.5)
    # Draw forward arrow from pelvis
    pelvis = g[0]
    ax.annotate("", xy=(pelvis[0], pelvis[2] + 0.25),
                xytext=(pelvis[0], pelvis[2]),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))
    ax.set_title(title, fontsize=8, pad=2)
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.4)

def frames_to_video(frames, path, fps=15):
    writer = imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8)
    for f in frames:
        writer.append_data(f)
    writer.close()
    print(f"  ✅ Saved: {path}")


# ================================================================== #
#  VIDEO 1: random_face  —  6 個 panel 俯視圖，同一sequence不同旋轉
# ================================================================== #
def make_video1_random_face(data_dir, out_dir, seq_idx=0):
    print("\n[Video 1] random_face: 6 rotations of the same sequence (top-down view)")

    # 取 disable_sliding 保證每個 pk 只有一個 window，
    # 用 data_ratio=0.005 只讀少量資料加速
    ds_base = FineFS(data_dir, input_n=30, output_n=40, split=1, mode="rotation",
                     move_global=True, random_face=False, reward_mode=True,
                     disable_sliding=True, data_ratio=0.02)

    if len(ds_base) == 0:
        print("  [SKIP] No data found.")
        return

    sample_idx = seq_idx % len(ds_base)
    # Ground-truth joints for reference (no rotation)
    base_joints = ds_base[sample_idx]["pose"][:3].permute(1, 2, 0).numpy()  # (T, V, 3)
    motion_name = ds_base[sample_idx]["motion_name"]
    score       = ds_base[sample_idx]["judge_score"].item()
    T = base_joints.shape[0]

    # Build 6 randomly rotated versions at dataset level
    ds_aug = FineFS(data_dir, input_n=30, output_n=40, split=1, mode="rotation",
                    move_global=True, random_face=True, reward_mode=True,
                    disable_sliding=True, data_ratio=0.02)

    N_ROT = 6
    rotated_seqs = []
    for _ in range(N_ROT):
        # getitem calls random_face each time → different theta each call
        rotated_seqs.append(ds_aug[sample_idx]["pose"][:3].permute(1, 2, 0).numpy())  # (T,V,3)

    # Render frame by frame
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f"Video 1: random_face test  |  motion={motion_name}  score={score:.2f}\n"
                 "Top-left: original (no rotation)   Others: 6 random rotations\n"
                 "✅ Expect same skeleton shape but different horizontal facing direction",
                 fontsize=9)

    all_axes = axes.flatten()  # 8 axes: [0]=original, [1-6]=rotations, [7]=timestamp

    frames = []
    for t in range(T):
        # Panel 0: original
        draw_skeleton_topdown(all_axes[0], base_joints[t], title="Original\n(random_face=False)", color="#555555")
        for i, seq in enumerate(rotated_seqs):
            ax = all_axes[i + 1]
            draw_skeleton_topdown(ax, seq[t], title=f"Rotation #{i+1}", color=PANEL_COLORS[i])
        # Panel 7: timestamp
        all_axes[7].clear()
        all_axes[7].text(0.5, 0.5, f"Frame\n{t+1}/{T}", ha="center", va="center",
                        fontsize=20, transform=all_axes[7].transAxes)
        all_axes[7].axis("off")

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
        frames.append(img.copy())

    plt.close(fig)
    frames_to_video(frames, Path(out_dir) / "video1_random_face.mp4")


# ================================================================== #
#  VIDEO 2: move_global  —  左右對比，global vs local
# ================================================================== #
def make_video2_move_global(data_dir, out_dir, seq_idx=0):
    print("\n[Video 2] move_global: global trajectory (left) vs local/root-centric (right)")

    ds_global = FineFS(data_dir, input_n=30, output_n=40, split=1, mode="rotation",
                       move_global=True,  random_face=False, reward_mode=True,
                       disable_sliding=True, data_ratio=0.02)
    ds_local  = FineFS(data_dir, input_n=30, output_n=40, split=1, mode="rotation",
                       move_global=False, random_face=False, reward_mode=True,
                       disable_sliding=True, data_ratio=0.02)

    if len(ds_global) == 0:
        print("  [SKIP] No data found.")
        return

    idx = seq_idx % len(ds_global)
    joints_global = ds_global[idx]["pose"][:3].permute(1, 2, 0).numpy()  # (T, V, 3)
    joints_local  = ds_local[idx]["pose"][:3].permute(1, 2, 0).numpy()
    motion_name   = ds_global[idx]["motion_name"]
    T = joints_global.shape[0]

    # Compute pelvis X-Z trajectory for global
    traj_x = joints_global[:, 0, 0]  # pelvis X over time
    traj_z = joints_global[:, 0, 2]  # pelvis Z over time

    fig = plt.figure(figsize=(15, 6))
    fig.suptitle(f"Video 2: move_global test  |  motion={motion_name}\n"
                 "Left: move_global=True (world coords, trajectory drifts)  "
                 "Centre: move_global=False (camera/local, stays near origin)  "
                 "Right: pelvis X-Z trajectory over time",
                 fontsize=9)
    ax_global = fig.add_subplot(1, 3, 1, projection='3d')  # 3D left
    ax_local  = fig.add_subplot(1, 3, 2, projection='3d')  # 3D centre
    ax_traj   = fig.add_subplot(1, 3, 3)                   # 2D right

    frames = []
    for t in range(T):
        draw_skeleton_3d(ax_global, joints_global[t],
                        title=f"move_global=True\n(t={t+1})", color="#e63946", elev=15, azim=45)
        draw_skeleton_3d(ax_local, joints_local[t],
                        title=f"move_global=False\n(t={t+1})", color="#457b9d", elev=15, azim=45)

        # Trajectory plot (right panel)
        ax_traj.clear()
        ax_traj.plot(traj_x[:t+1], traj_z[:t+1], c="#e63946", lw=2, label="global pelvis")
        ax_traj.scatter([traj_x[t]], [traj_z[t]], c="#e63946", s=60)
        ax_traj.axhline(0, color="gray", lw=0.5, ls="--")
        ax_traj.axvline(0, color="gray", lw=0.5, ls="--")
        ax_traj.set_xlim([-2, 2])
        ax_traj.set_ylim([-2, 2])
        ax_traj.set_aspect("equal")
        ax_traj.set_title("Pelvis top-down trajectory\n(global)", fontsize=8)
        ax_traj.set_xlabel("X"); ax_traj.set_ylabel("Z")
        ax_traj.grid(True, linestyle="--", alpha=0.4)
        ax_traj.legend(fontsize=7)

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
        frames.append(img.copy())

    plt.close(fig)
    frames_to_video(frames, Path(out_dir) / "video2_move_global.mp4")



# ================================================================== #
#  VIDEO 3: reward_mode vs main_mode  —  幾何一致性驗證
# ================================================================== #
def make_video3_mode_consistency(data_dir, out_dir, seq_idx=0):
    print("\n[Video 3] reward_mode vs main_mode: geometry consistency check")

    ds_reward = FineFS(data_dir, input_n=30, output_n=40, split=1, mode="rotation",
                       move_global=True, random_face=False, reward_mode=True,
                       disable_sliding=True, data_ratio=0.02)
    ds_main   = FineFS(data_dir, input_n=30, output_n=40, split=1, mode="rotation",
                       move_global=True, random_face=False, reward_mode=False,
                       disable_sliding=True, data_ratio=0.02)

    if len(ds_reward) == 0:
        print("  [SKIP] No data found.")
        return

    idx = seq_idx % len(ds_reward)

    # reward_mode: pose (6, T, V) — first 3 channels are joint XYZ
    reward_joints = ds_reward[idx]["pose"][:3].permute(1, 2, 0).numpy()   # (T, V, 3)

    # main_mode: pose (T, V*3) ndarray
    main_pose_flat = ds_main[idx]["pose"]   # (T, 72)
    T, D = main_pose_flat.shape
    V = D // 3
    main_joints = main_pose_flat.reshape(T, V, 3)                          # (T, V, 3)

    motion_name = ds_reward[idx]["motion_name"]

    # Compute per-frame per-joint difference
    diffs = np.abs(reward_joints - main_joints)   # (T, V, 3)
    max_diff = diffs.max()
    mean_diff = diffs.mean()
    print(f"  Max joint coord difference: {max_diff:.6f}  Mean: {mean_diff:.6f}")
    if max_diff < 1e-4:
        consistency_msg = f"CONSISTENT  (max_diff={max_diff:.2e})"
        msg_color = "green"
    else:
        consistency_msg = f"INCONSISTENT!  (max_diff={max_diff:.4f})"
        msg_color = "red"
    print(f"  {consistency_msg}")

    fig = plt.figure(figsize=(15, 6))
    fig.suptitle(f"Video 3: mode consistency  |  motion={motion_name}\n"
                 "Left: reward_mode=True   Centre: reward_mode=False   Right: coord error\n"
                 f"{consistency_msg}",
                 fontsize=9, color=msg_color)
    ax_r  = fig.add_subplot(1, 3, 1, projection='3d')  # 3D reward
    ax_m  = fig.add_subplot(1, 3, 2, projection='3d')  # 3D main
    ax_e  = fig.add_subplot(1, 3, 3)                   # 2D error chart

    # Error time series data
    per_frame_max = diffs.max(axis=(1, 2))   # (T,)

    frames = []
    for t in range(T):
        draw_skeleton_3d(ax_r, reward_joints[t],
                        title=f"reward_mode=True\n(t={t+1})", color="#2a9d8f")
        draw_skeleton_3d(ax_m, main_joints[t],
                        title=f"reward_mode=False\n(t={t+1})", color="#e9c46a")

        # Error chart
        ax_e.clear()
        ax_e.plot(per_frame_max, color="#e63946", lw=1.5, label="max joint err")
        ax_e.axvline(t, color="black", lw=1, ls="--", alpha=0.5)
        ax_e.axhline(1e-4, color="green", lw=1, ls=":", label="threshold 1e-4")
        ax_e.set_xlim([0, T])
        ax_e.set_ylim([0, max(per_frame_max.max() * 1.1, 1e-3)])
        ax_e.set_title("Max joint coord error per frame", fontsize=8)
        ax_e.set_xlabel("frame"); ax_e.set_ylabel("max |delta|")
        ax_e.legend(fontsize=7)
        ax_e.grid(True, linestyle="--", alpha=0.4)

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
        frames.append(img.copy())

    plt.close(fig)
    frames_to_video(frames, Path(out_dir) / "video3_mode_consistency.mp4")



# ================================================================== #
#  Main
# ================================================================== #
def main():
    parser = argparse.ArgumentParser(description="FineFS visual verification tool")
    parser.add_argument("--data_dir", type=str,
                        default="/home/allen/datasets/FineFS_5s/3_final")
    parser.add_argument("--out_dir", type=str,
                        default=str(Path(__file__).parent / "finefs_verify_output"),
                        help="Output directory for verification videos")
    parser.add_argument("--seq_idx", type=int, default=3,
                        help="Which sequence index to use (among loaded sequences)")
    parser.add_argument("--skip", type=str, default="",
                        help="Comma-separated video numbers to skip, e.g. '2,3'")
    args = parser.parse_args()

    skip = set(args.skip.split(",")) if args.skip else set()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Output dir: {args.out_dir}")
    print(f"Data dir  : {args.data_dir}")
    print(f"Seq index : {args.seq_idx}")

    if "1" not in skip:
        make_video1_random_face(args.data_dir, args.out_dir, args.seq_idx)
    if "2" not in skip:
        make_video2_move_global(args.data_dir, args.out_dir, args.seq_idx)
    if "3" not in skip:
        make_video3_mode_consistency(args.data_dir, args.out_dir, args.seq_idx)

    print(f"\n🎬 All done. Videos saved to: {args.out_dir}")
    print("  video1_random_face.mp4    — verify random_face Y-axis rotation")
    print("  video2_move_global.mp4    — verify move_global trajectory difference")
    print("  video3_mode_consistency.mp4 — verify reward_mode vs main_mode are identical")


if __name__ == "__main__":
    main()
