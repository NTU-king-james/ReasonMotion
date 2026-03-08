
import os
import sys
import pickle
import argparse
import numpy as np
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from utils.finefs import EDGES

def calculate_rotation(pose):
    """
    Calculate total rotation turns based on Hip joints.
    Assumes standard skeleton where:
    Index 1: Left Hip
    Index 2: Right Hip
    Coordinate system: Y is vertical (based on visualization logic where Vis Z = -Data Y)
    Rotation Plane: X-Z
    """
    # pose shape: (T, 24, 3)
    
    # Vector from Left Hip (1) to Right Hip (2)
    left_hip = pose[:, 1, :]
    right_hip = pose[:, 2, :]
    
    vec = right_hip - left_hip # (T, 3)
    
    # [Mod] Vector Smoothing (Kernel=5) to remove jitter
    # Using simple moving average on T dimension
    kernel_size = 5
    kernel = np.ones(kernel_size) / kernel_size
    
    # Pad edges to keep shape same size
    # logical equivalent to 'same' convolution but we want to avoid boundary artifacts being too strong
    # For T=60, edge effects are minimal.
    # We apply smoothing to x and z components
    
    vec_x_raw = vec[:, 0]
    vec_z_raw = vec[:, 2]
    
    # Mode 'same' returns output of length max(M, N). Boundary effects handle by zero-padding in numpy convolve?
    # Actually np.convolve mode='same' is good.
    # To match pytorch avg_pool1d with padding=2, stride=1:
    # It essentially averages a window.
    
    vec_x = np.convolve(vec_x_raw, kernel, mode='same')
    vec_z = np.convolve(vec_z_raw, kernel, mode='same')
    
    # Calculate angles in radians (-pi to pi)
    angles = np.arctan2(vec_z, vec_x)
    
    # Calculate frame-to-frame difference
    deltas = angles[1:] - angles[:-1]
    
    # Unwrap phase
    # (delta + pi) % (2pi) - pi
    deltas_unwrapped = (deltas + np.pi) % (2 * np.pi) - np.pi
    
    # [Mod] Total Turns Calculation
    # Old: sum(abs(delta)) -> path length
    # New: abs(sum(delta)) -> net displacement
    # Rationale: User wants "signed sum then absolute" to capture net rotation
    # and cancel out jitter (+1, -1).
    
    total_angle = np.abs(np.sum(deltas_unwrapped))
    total_turns = total_angle / (2 * np.pi)
    
    return total_turns, angles

def render_video(pkl_path, output_path):
    print(f"📂 Loading {pkl_path}...")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    
    # Handle different key names
    if 'pred_xyz_24_struct_global' in data:
        key = 'pred_xyz_24_struct_global'
    elif 'pred_xyz_24_struct' in data:
        key = 'pred_xyz_24_struct'
    else:
        # Fallback: maybe it's just the array?
        key = list(data.keys())[0]
        print(f"⚠️ Unknown key structure. Using key: {key}")
        
    xyz = data[key].astype(np.float32) # (T, 72) or (T, 24, 3) ?
    
    # Check shape
    if len(xyz.shape) == 2 and xyz.shape[1] == 72:
        xyz = xyz.reshape(-1, 24, 3)
    elif len(xyz.shape) == 3 and xyz.shape[1] == 24 and xyz.shape[2] == 3:
        pass
    else:
        raise ValueError(f"Unexpected shape: {xyz.shape}")
        
    T = xyz.shape[0]
    print(f"ℹ️ Sequence Length: {T} frames")
    
    # Calculate Rotation
    total_turns, angles = calculate_rotation(xyz)
    print(f"🔄 Calculated Total Turns: {total_turns:.4f}")
    
    # Setup Plot
    frames = []
    
    # Bounds logic (from visualize_every_batch.py)
    min_vals = np.min(xyz, axis=(0, 1))
    max_vals = np.max(xyz, axis=(0, 1))
    bound = 0.8 # Slightly larger for safety
    
    # Mapping: Vis X = Data X, Vis Y = Data Z (Depth), Vis Z = -Data Y (Height)
    # Start with Data X
    x_lim = [min(-bound, min_vals[0]), max(bound, max_vals[0])]
    # Data Z
    z_lim = [min(-bound, min_vals[2]), max(bound, max_vals[2])]
    # Data Y (Height). Typically -0.5 to 0.5 range roughly
    # In Vis Z, it will be -Data Y. 
    # If Data Y is roughly [-1, 1], Vis Z should be capable.
    # Let's check max vals of Y.
    # Fixed Y limit seems preferred in original script: [-0.5, 0.5]
    y_lim = [-0.8, 0.8] 

    lx = x_lim[1] - x_lim[0]
    ly = z_lim[1] - z_lim[0]
    lz = y_lim[1] - y_lim[0]
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect((lx, ly, lz))
    
    print("🎥 Rendering frames...")
    for t in tqdm(range(T)):
        ax.clear()
        
        ax.set_xlim(x_lim)
        ax.set_ylim(z_lim) 
        ax.set_zlim(y_lim)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Z (Depth)')
        ax.set_zlabel('Y (Height)')
        
        # Determine current accumulated turns for dynamic display?
        # Or just total. User asked "Total turns calculated".
        # Let's show Total Turns static, and maybe Current Turn dynamic?
        # Let's just show Total as requested.
        
        ax.set_title(f"Check Rotation | Frame {t}/{T}")
        
        # Add text
        # transform=ax.transAxes puts it in relative coords (0,0 bottom-left, 1,1 top-right)
        ax.text2D(0.95, 0.95, f"Total Turns: {total_turns:.2f}", transform=ax.transAxes, 
                  color='red', fontsize=14, fontweight='bold', ha='right')
        
        # Camera
        ax.view_init(elev=10, azim=-90)
        ax.dist = 8.0
        
        pose = xyz[t]
        
        # Draw Skeleton
        xs = pose[:, 0]
        ys = pose[:, 2]   # Data Z -> Vis Y
        zs = -pose[:, 1]  # -Data Y -> Vis Z
        
        # Points
        ax.scatter(xs, ys, zs, c='black', s=20, alpha=0.6)
        
        # Edges
        for (v1, v2) in EDGES:
            x_pair = [xs[v1], xs[v2]]
            y_pair = [ys[v1], ys[v2]]
            z_pair = [zs[v1], zs[v2]]
            ax.plot(x_pair, y_pair, z_pair, color='blue', linewidth=2)
            
        # Draw Heading Vector (Left Hip -> Right Hip) to visualize rotation logic
        lh_idx, rh_idx = 1, 2
        hx = [xs[lh_idx], xs[rh_idx]]
        hy = [ys[lh_idx], ys[rh_idx]]
        hz = [zs[lh_idx], zs[rh_idx]]
        ax.plot(hx, hy, hz, color='red', linewidth=3, label='Hip Vector')
        
        fig.canvas.draw()
        try:
            image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            image = image[:, :, :3].copy()
            frames.append(image)
        except Exception as e:
            print(f"Error capturing frame: {e}")
            break
            
    plt.close(fig)
    
    if frames:
        imageio.mimsave(output_path, frames, fps=30)
        print(f"✅ Video saved to: {output_path}")
    else:
        print("❌ No frames generated.")

if __name__ == "__main__":
    """
    python utils/debug_tools/visualize_rotation.py \
        --pkl_path /home/allen/datasets/FineFS_5s/3_final/test/3Lz/3Lz_0006/new_res.pk
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl_path", type=str, help="Path to the skeleton .pkl file")
    parser.add_argument("--output", type=str, default="utils/debug_tools/rotation_check.mp4", help="Output video path")
    args = parser.parse_args()
    
    if not os.path.exists(args.pkl_path):
        print(f"❌ File not found: {args.pkl_path}")
        sys.exit(1)
        
    render_video(args.pkl_path, args.output)
