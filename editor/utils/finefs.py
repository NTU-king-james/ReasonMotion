# finefs.py  -- for Motion-Editing Dataset
import sys, os, glob, pickle, random, json
from typing import Union
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from typing import List, Dict, Optional, Any
from torch.utils.data import Dataset
from collections import deque

# ★ Switch to "Unified Version" motion editor ★
from Editor.motion_editor_all import (
    process_command,           # Generate reverse error motion
    J,                         # Joint index mapping
    descend as get_descendants,   # Get joint descendant chain
    simple_edges as connectivity,  # Skeleton edge list for drawing
    expand_commands
)

def build_tree(edge_list):
    tree = {}
    for a, b in edge_list:
        tree.setdefault(a, []).append(b)
    return tree

# ========== Visualization tools (Retain from old version) ================================
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, imageio

def sequence_to_video(orig, edited, out_path, bones, fps=30):
    """Save two sequences as mp4 with gray / green contrast for debugging"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig = plt.figure(figsize=(6,6))
    ax  = fig.add_subplot(111, projection='3d')
    writer = imageio.get_writer(str(out_path), fps=fps, codec='libx264', bitrate='16M')
    for o,e in zip(orig, edited):
        ax.clear()
        ax.scatter(o[:,0],o[:,2],-o[:,1],c='gray',s=15)
        ax.scatter(e[:,0],e[:,2],-e[:,1],c='lime',s=15)
        for s,t in bones:
            ax.plot([o[s,0],o[t,0]],[o[s,2],o[t,2]],[-o[s,1],-o[t,1]],c='gray',lw=1)
            ax.plot([e[s,0],e[t,0]],[e[s,2],e[t,2]],[-e[s,1],-e[t,1]],c='lime',lw=1)
        ax.view_init(20,45)
        ax.set(xlim=[-0.5,0.5],ylim=[-0.5,0.5],zlim=[-0.5,0.5])
        fig.canvas.draw(); writer.append_data(np.asarray(fig.canvas.renderer.buffer_rgba())[:,:,:3])
    writer.close(); plt.close(fig)

# ========================== FineFS Dataset =========================
class FineFS(Dataset):
    """
    Transform original pose into "reverse error motion" according to json instruction,
    Generate (pose, pose_edit, mask, timepoints, command) for model training.
    """
    def __init__(
        self,
        data_dir: str,
        seq_len: int,
        split: int = 0,
        data_ratio: float = 1.0,          # 0~1
        move_global: bool = True,         # reserved parameter, not used for now
        all_mask: bool = False,
        edit_ratio: float = 1.0,          # sampling ratio for command
        command_json_path: Union[str, List[str]] = None,
        use_magnitude: bool = False,      # produce slightly / completely version
        mix_motion: bool = False          # True  -> can mix torso/hand/foot
    ):
        super().__init__()
        self.seq_len     = seq_len
        self.all_mask    = all_mask
        self.edit_ratio  = edit_ratio
        self.data_dir    = data_dir
        self.tree = build_tree(connectivity())

        # ------------- Read three instruction pools -------------
        if command_json_path is None:
            raise ValueError("Must specify command_json_path")
        if isinstance(command_json_path, str):
            command_json_path = [command_json_path]

        pools: dict[str, list] = {"hand": [], "foot": [], "torso": []}
        for p in command_json_path:
            with open(p, encoding="utf-8") as f:
                raw_cmds = json.load(f)
            cmd_pool = expand_commands(raw_cmds, use_magnitude=use_magnitude)
            key = ("hand"  if "hand"  in p
                   else "foot" if ("foot" in p or "leg" in p)
                   else "torso")
            pools[key].extend(cmd_pool)
            print(f"[CMD] loaded {len(cmd_pool):>3} cmds from {p}")

        # Flatten complete pool
        self.motion_commands = [*pools["hand"], *pools["foot"], *pools["torso"]]
        self.n_cmds = len(self.motion_commands)
        print(f"[CMD] total pooled commands : {self.n_cmds}")

        # ------------- motion file list -------------
        split_name = ["train", "valid", "test"][split]
        all_files  = glob.glob(os.path.join(data_dir, split_name, "*", "*", "new_res.pk"))
        if data_ratio < 1.0:
            all_files = random.sample(all_files, int(len(all_files) * data_ratio))
        if not all_files:
            raise FileNotFoundError(f"No .pk files in {data_dir}/{split_name}")
        self.pk_files = sorted(all_files)
        print(f"[{split_name}] use {len(self.pk_files)} motion files")

        # ------------- Generate samples -------------
        self.all_samples: list[dict] = []
        from collections import defaultdict
        text_counter = defaultdict(int)

        for pk in self.pk_files:
            with open(pk, "rb") as f:
                arr = pickle.load(f)["pred_xyz_24_struct_global"].astype(np.float32)
            if arr.shape[0] < seq_len:
                continue
            src_pose = arr[:seq_len].reshape(seq_len, 24, 3)

            # Determine how many groups of samples to generate for this video
            n_edit = max(1, int(round(self.edit_ratio * self.n_cmds)))

            if not mix_motion:
                # ------ Single instruction version ------
                for cmd in random.sample(self.motion_commands, n_edit):
                    pose_edit = process_command(src_pose, cmd, use_magnitude)
                    mask      = self._build_mask(cmd, seq_len)
                    self._push_sample(src_pose, pose_edit, mask, cmd["command"])
                    text_counter[cmd["command"]] += 1
            else:
                # ------ Mixed version: Randomly sample 1~3 categories per loop ------
                for _ in range(n_edit):
                    k = random.randint(1, 3)                             # 1~3 pool
                    categories = random.sample(["torso", "hand", "foot"], k)

                    ordered_cmds = []
                    for cat in ("torso", "hand", "foot"):                # Fixed apply order
                        if cat in categories and pools[cat]:
                            ordered_cmds.append(random.choice(pools[cat]))
                    if not ordered_cmds:           # <- **Key: skip if no motion**
                        continue
                    # Apply multiple motions
                    pose_edit = src_pose.copy()
                    for cmd in ordered_cmds:
                        pose_edit = process_command(pose_edit, cmd, use_magnitude)

                    # Merge text
                    joint_text = " and ".join(c["command"] for c in ordered_cmds)

                    # Merge mask (union: if there is any 0, set to 0)
                    mask = np.zeros((seq_len, 24*3), np.float32) if self.all_mask else np.ones((seq_len, 24*3), np.float32)
                    #mask = np.ones((seq_len, 24*3), np.float32)
                    if not self.all_mask:
                        for cmd in ordered_cmds:
                            mask  = np.minimum(mask, self._build_mask(cmd, seq_len))
                    print(f"mask =",mask[0:3,:].mean(axis=0))  # debug mask
                    self._push_sample(src_pose, pose_edit, mask, joint_text)
                    text_counter[joint_text] += 1

        # ───────────── Debug summary ─────────────
        print("\n========== Debug Info ==========")
        print(f"Dataset length                       : {len(self.all_samples)}")
        print(f"Loaded motion files                  : {len(self.pk_files)}")
        print(f"seq_len                              : {self.seq_len}")
        print(f"edit_ratio                           : {self.edit_ratio}")
        print(f"all_mask                             : {self.all_mask}")
        print(f"use_magnitude                        : {use_magnitude}")
        print(f"mix_motion                           : {mix_motion}")
        print("Top-20 text frequency:")
        for txt, cnt in sorted(text_counter.items(), key=lambda x: -x[1])[:20]:
            print(f"  {cnt:>5}  | {txt}")
        print("====================================\n")

    # ------------------ helper: generate mask ------------------
    def _build_mask(self, cmd: dict, T: int) -> np.ndarray:
        mask = np.ones((T, 24*3), np.float32)
        if self.all_mask:
            mask[:] = 0
            #print("mask = ", mask[0:3,:].mean(axis=0))  # debug mask
        else:
            tgt_joint = J[cmd["target_joint"]]
            propagate = cmd.get("propagate", True)
            chain = get_descendants(self.tree, tgt_joint) if propagate else [tgt_joint]

            for j in chain:
                mask[:, j*3:j*3+3] = 0.0
        return mask

    # ------------------ helper: push sample ---------------
    def _push_sample(self,
                     pose: np.ndarray,
                     pose_edit: np.ndarray,
                     mask: np.ndarray,
                     cmd_text: str):
        self.all_samples.append({
            "pose":        pose.reshape(self.seq_len, 24*3),
            "pose_edit":   pose_edit.reshape(self.seq_len, 24*3),
            "mask":        mask,
            "timepoints":  np.arange(self.seq_len),
            "command":     cmd_text
        })



    # ----------- Standard Dataset interface -----------
    def __len__(self): return len(self.all_samples)
    def __getitem__(self, idx): return self.all_samples[idx]

# ---------------- Quick sanity check -------------------------------
if __name__ == "__main__":
    ds = FineFS(
        data_dir="/home/allen/datasets/FineFS_5s/3_final",
        seq_len=30,
        split=0,
        data_ratio=0.002,
        all_mask=True,
        edit_ratio=0.3,
        command_json_path=[
            "/home/allen/Diffusion/DePOSit_Skating_Editor_BaseModel/Editor/two_hands_commands.json",
            "/home/allen/Diffusion/DePOSit_Skating_Editor_BaseModel/Editor/leg/two_feet_commands.json",
            "/home/allen/Diffusion/DePOSit_Skating_Editor_BaseModel/Editor/body/body_commands.json"
        ]
    )

    idx = random.randint(0,len(ds)-1)
    sample = ds[idx]
    print("cmd :", sample["command"])
    sequence_to_video(
        sample["pose"].reshape(30,24,3),
        sample["pose_edit"].reshape(30,24,3),
        "/home/allen/Diffusion/DePOSit_Skating_Editor_BaseModel/debug_pose.mp4",
        connectivity()
    )
    print("saved demo video -> debug_pose.mp4")
