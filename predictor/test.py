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
import numpy as np
from scipy.spatial.distance import pdist, squareform
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from model.model import ModelMain
from motion_data.finefs import FineFS
from motion_data.h36m_unified import H36MUnified
from utils.metrics import (
    MetricsEvaluator,
    ampjpe,
    fmpjpe,
    compute_ade,
    compute_fde,
    compute_mmade,
    compute_mmfde,
    compute_diversity,
)
from utils.text_encoder import TextEncoder
from tqdm import tqdm

def load_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def resolve_config_path(ckpt_path, config_path=None, fallback="configs/train_grpo.yaml"):
    if config_path is not None:
        return config_path

    derived_config = os.path.join(os.path.dirname(ckpt_path), "..", "config.yaml")
    if os.path.exists(derived_config):
        print(f"found config file at {derived_config}")
        return derived_config

    print(f"config file not found at {derived_config}, using default: {fallback}")
    return fallback


def infer_target_dim(config):
    if "model" in config and "target_dim" in config["model"]:
        return config["model"]["target_dim"]
    ds_name = config["data"].get("dataset", config["data"].get("name", "h36m")).lower()
    if ds_name == "h36m":
        return 17 * 3
    return 24 * 3


def load_model_checkpoint(model, ckpt_path, device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"📥 Loading checkpoint: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device)
    if isinstance(state_dict, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in state_dict and isinstance(state_dict[key], dict):
                state_dict = state_dict[key]
                break

    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported checkpoint format: {ckpt_path}")

    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    try:
        model.load_state_dict(new_state_dict)
        print("✅ Checkpoint loaded successfully.")
    except Exception:
        print("⚠️ Strict loading failed, trying strict=False...")
        model.load_state_dict(new_state_dict, strict=False)
        print("✅ Checkpoint loaded (strict=False).")


def build_model_from_checkpoint(config, ckpt_path, device):
    target_dim = infer_target_dim(config)
    model = ModelMain(config, device=device, target_dim=target_dim).to(device)
    load_model_checkpoint(model, ckpt_path, device)
    model.eval()
    return model


def repeat_text_embedding(text_embedding, repeats):
    if text_embedding is None:
        return None
    if isinstance(text_embedding, tuple):
        return tuple(x.repeat_interleave(repeats, dim=0) for x in text_embedding)
    return text_embedding.repeat_interleave(repeats, dim=0)


def repeat_batch_for_samples(batch, repeats):
    repeated = {}
    batch_size = None
    for value in batch.values():
        if isinstance(value, torch.Tensor):
            batch_size = value.shape[0]
            break

    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            repeated[key] = value.repeat_interleave(repeats, dim=0)
        elif batch_size is not None and isinstance(value, (list, tuple)) and len(value) == batch_size:
            repeated[key] = [item for item in value for _ in range(repeats)]
        else:
            repeated[key] = value
    return repeated


class TwoStageEvaluatorModel(torch.nn.Module):
    """Wrap two predictor models behind the same evaluate API used by MetricsEvaluator."""
    def __init__(self, model_s, model_l, config_l, n1, device):
        super().__init__()
        self.model_s = model_s
        self.model_l = model_l
        self.input_n = config_l["data"]["input_n"]
        self.n1 = n1
        self.device = device

        if self.n1 <= 0:
            raise ValueError("--two_stage_n1 must be > 0")
        if self.n1 >= self.input_n:
            raise ValueError("--two_stage_n1 must be smaller than model_l input_n")

    def eval(self):
        super().eval()
        self.model_s.eval()
        self.model_l.eval()
        return self

    @torch.no_grad()
    def evaluate(self, batch, n_samples, text_embedding=None, noisy_data=None, sample=True):
        short_samples, gt, missing_mask, tp = self.model_s.evaluate(
            batch,
            n_samples,
            text_embedding=text_embedding,
            noisy_data=noisy_data,
            sample=sample,
        )

        if short_samples.dim() != 4:
            raise ValueError(f"Expected short samples with shape (B, N, K, T), got {short_samples.shape}")

        pose = batch["pose"].to(self.device).float()
        mask = batch["mask"].to(self.device).float()
        timepoints = batch["timepoints"].to(self.device).float()

        batch_size, n_branch, k_dim, total_len = short_samples.shape
        if pose.shape[1] != total_len:
            raise ValueError(
                f"Batch pose length ({pose.shape[1]}) and generated length ({total_len}) do not match."
            )
        if self.input_n + self.n1 > total_len:
            raise ValueError(
                f"input_n + two_stage_n1 ({self.input_n + self.n1}) exceeds sequence length ({total_len})."
            )

        pose_rep = pose.unsqueeze(1).expand(batch_size, n_branch, total_len, k_dim)
        pose_rep = pose_rep.reshape(batch_size * n_branch, total_len, k_dim).clone()
        mask_rep = mask.unsqueeze(1).expand(batch_size, n_branch, total_len, k_dim)
        mask_rep = mask_rep.reshape(batch_size * n_branch, total_len, k_dim).clone()
        short_flat = short_samples.permute(0, 1, 3, 2).reshape(batch_size * n_branch, total_len, k_dim)

        stage2_pose = pose_rep.clone()
        stage2_pose[:, :total_len - self.n1] = pose_rep[:, self.n1:]
        stage2_pose[:, total_len - self.n1:] = pose_rep[:, -1:].expand(-1, self.n1, -1)
        stage2_pose[:, self.input_n - self.n1:self.input_n] = short_flat[
            :, self.input_n:self.input_n + self.n1
        ]

        stage2_mask = torch.zeros_like(mask_rep)
        stage2_mask[:, :self.input_n - self.n1] = mask_rep[:, self.n1:self.input_n]
        stage2_mask[:, self.input_n - self.n1:self.input_n] = 1.0

        stage2_batch = repeat_batch_for_samples(batch, n_branch)
        stage2_batch["pose"] = stage2_pose
        stage2_batch["mask"] = stage2_mask
        stage2_batch["timepoints"] = timepoints.repeat_interleave(n_branch, dim=0)
        stage2_text = repeat_text_embedding(text_embedding, n_branch)

        long_samples = self.model_l.evaluate(
            stage2_batch,
            1,
            text_embedding=stage2_text,
            noisy_data=None,
            sample=sample,
        )[0]

        long_flat = long_samples[:, 0]
        final_flat = pose_rep.permute(0, 2, 1).clone()
        short_flat_kt = short_flat.permute(0, 2, 1)
        final_flat[:, :, self.input_n:self.input_n + self.n1] = short_flat_kt[
            :, :, self.input_n:self.input_n + self.n1
        ]
        final_flat[:, :, self.input_n + self.n1:] = long_flat[:, :, self.input_n:total_len - self.n1]

        final_samples = final_flat.reshape(batch_size, n_branch, k_dim, total_len)
        gt = pose.permute(0, 2, 1)
        missing_mask = 1 - mask.permute(0, 2, 1)
        return final_samples, gt, missing_mask, timepoints


class IndexedDataset(Dataset):
    """Attach sample_idx to each item so evaluation can map to precomputed multi-GT."""
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        if isinstance(sample, dict):
            out = dict(sample)
            out["sample_idx"] = idx
            return out
        return sample


def build_h36m_multimodal_gt(dataset, threshold):
    """Build multi-GT sets by nearest history-end pose, mirroring CoMusion protocol."""
    if not isinstance(dataset, H36MUnified):
        return None

    n = len(dataset.data_idx)
    print(f"Building multi-GT for H36M samples: {n} (threshold={threshold})")

    futures = []
    start_poses = []
    for sample_idx in range(n):
        # key 是原始影片，sample_idx 是經過 slide window 後的 index
        key, start_frame = dataset.data_idx[sample_idx]
        # frame start
        fs = np.arange(start_frame, start_frame + dataset.seq_len)
        pose_32 = dataset.p3d[key][fs]
        pose = (pose_32.copy() / 1000.0)[:, dataset.dim_used]
        # 利用最後一個 frame 作比較，與 Comusion 相同
        start_poses.append(pose[dataset.in_n - 1])
        futures.append(pose[dataset.in_n:])

    start_poses = np.asarray(start_poses, dtype=np.float64)
    futures = np.asarray(futures, dtype=np.float32)

    # Match CoMusion behavior: full pairwise distances via scipy pdist + squareform.
    pd = squareform(pdist(start_poses))
    multimodal_traj = []
    for i in range(pd.shape[0]):
        ind = np.nonzero(pd[i] < threshold)[0]
        multimodal_traj.append(torch.from_numpy(futures[ind]))

    valid = sum(1 for x in multimodal_traj if x.shape[0] > 1)
    print(f"multi-GT ready: valid multi-modal samples={valid}/{n}")
    return multimodal_traj

@torch.no_grad()
def evaluate_with_multimodal_gt(model, dataloader, config, device, nsample, multimodal_traj):
    """Evaluate metrics with true multi-GT (MMADE/MMFDE skip n_gts==1 as NaN)."""
    model.eval()
    target_dim = config['model'].get('target_dim', 72)
    input_n = config['data'].get('input_n', 0)
    text_encoder = TextEncoder(device=str(device))

    amp_total, fmp_total = 0.0, 0.0
    ade_total, fde_total = 0.0, 0.0
    diversity_total = 0.0
    mmade_vals, mmfde_vals = [], []
    mm_valid_count = 0
    mmade_running_sum, mmfde_running_sum = 0.0, 0.0
    n_sequences = 0

    pbar = tqdm(dataloader, desc="Evaluating (multi-GT)")
    for batch in pbar:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        t_emb = text_encoder(batch.get("motion_name"))

        try:
            samples, gt = model.evaluate(batch, nsample, t_emb)[:2]  # (B, N, K, T)
            if gt.dim() == 3:
                gt = gt.unsqueeze(1)
            gt = gt.expand_as(samples)

            p_part, g_part = samples[..., input_n:], gt[..., input_n:]
            batch_size = p_part.shape[0]
            amp_total += float(ampjpe(p_part, g_part, target_dim).item()) * batch_size
            fmp_total += float(fmpjpe(p_part, g_part, target_dim).item()) * batch_size

            pred_np = p_part.cpu().numpy().transpose(0, 1, 3, 2)  # (B, N, T, K)
            gt_np = g_part.cpu().numpy().transpose(0, 1, 3, 2)    # (B, N, T, K)

            sample_indices = batch["sample_idx"]
            if isinstance(sample_indices, torch.Tensor):
                sample_indices = sample_indices.cpu().tolist()

            for b in range(pred_np.shape[0]):
                pred_b = pred_np[b]
                gt_b = gt_np[b, 0]
                sample_idx = int(sample_indices[b])

                ade_total += float(compute_ade(pred_b, gt_b))
                fde_total += float(compute_fde(pred_b, gt_b))
                diversity_total += float(compute_diversity(pred_b))

                if nsample > 1:
                    gt_multi = multimodal_traj[sample_idx].cpu().numpy()
                    if gt_multi.shape[0] <= 1:
                        mmade_vals.append(np.nan)
                        mmfde_vals.append(np.nan)
                    else:
                        mmade_i = float(compute_mmade(pred_b, gt_b, gt_multi))
                        mmfde_i = float(compute_mmfde(pred_b, gt_b, gt_multi))
                        mmade_vals.append(mmade_i)
                        mmfde_vals.append(mmfde_i)
                        mmade_running_sum += mmade_i
                        mmfde_running_sum += mmfde_i
                        mm_valid_count += 1

                n_sequences += 1
        except Exception:
            continue

        postfix = {
            "ADE": f"{(ade_total / max(1, n_sequences)):.4f}",
            "FDE": f"{(fde_total / max(1, n_sequences)):.4f}",
        }
        if nsample > 1:
            if mm_valid_count > 0:
                postfix["MMADE"] = f"{(mmade_running_sum / mm_valid_count):.4f}"
                postfix["MMFDE"] = f"{(mmfde_running_sum / mm_valid_count):.4f}"
            else:
                postfix["MMADE"] = "NaN"
                postfix["MMFDE"] = "NaN"
        pbar.set_postfix(postfix)

    denom = max(1, n_sequences)
    mmade_avg = float(np.nanmean(mmade_vals)) if nsample > 1 and len(mmade_vals) > 0 else None
    mmfde_avg = float(np.nanmean(mmfde_vals)) if nsample > 1 and len(mmfde_vals) > 0 else None

    return {
        "AMPJPE": amp_total / denom,
        "FMPJPE": fmp_total / denom,
        "ADE": ade_total / denom,
        "FDE": fde_total / denom,
        "MMADE": mmade_avg,
        "MMFDE": mmfde_avg,
        "Diversity": diversity_total / denom,
    }

# 讀 config

def build_dataset(cfg, split):
    """依 cfg['data']['dataset'] 回傳對應 Dataset 物件"""
    ds_name   = cfg['data'].get('dataset', 'h36m').lower()
    print(f"Building dataset: {ds_name} (split={split})")
    common_kw = dict(
        input_n    = cfg['data']['input_n'],
        output_n   = cfg['data']['output_n'],
        skip_rate  = cfg['data'].get('skip_rate', 20),
        split      = split,
        max_len    = cfg['data'].get('max_len'),
    )
    common_kw['output_n'] = 100  # 驗證集只取 1/10，快速驗證
    if split == 2:
        common_kw['skip_rate'] = 20  # 驗證集只取 1/10，快速驗證
    if ds_name == 'finefs':
        return FineFS(
            data_dir = cfg['data']['data_dir'],
            mode     = cfg['data'].get('mode', 'full_name'),
            **common_kw
        ), 24 * 3
    elif ds_name == 'h36m':
        joints = 17
        return H36MUnified(
            data_dir=cfg['data']['data_dir'],
            joints=joints,
            downsample=cfg['data'].get('downsample', 1),
            no_overlap=cfg['data'].get('no_overlap', False),
            protocol=cfg['data'].get('h36m_protocol', 'predictor'),
            miss_type=cfg['data'].get('miss_type', 'no_miss'),
            miss_rate=cfg['data'].get('miss_rate', 0.2),
            all_data=cfg['data'].get('all_data', True),
            data_ratio=cfg['data'].get('data_ratio', 1.0),
            pad_short_sequences=cfg['data'].get('pad_short_sequences', False),
            **common_kw,
        ), joints * 3
    else:
        raise ValueError(f"Unknown dataset: {ds_name}")
    
def main():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint file")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--nsample", type=int, default=1, help="Number of samples generated per input")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--enable_multigt", action="store_true", help="Enable H36M multi-GT MMADE/MMFDE evaluation")
    parser.add_argument("--multimodal_threshold", type=float, default=None, help="Distance threshold for H36M multi-GT grouping")
    parser.add_argument("--eval_mode", choices=["single", "two_stage"], default="single", help="Evaluation generation mode")
    parser.add_argument("--model_s", type=str, default=None, help="Short-term checkpoint for two-stage evaluation")
    parser.add_argument("--model_l", type=str, default=None, help="Long-term checkpoint for two-stage evaluation")
    parser.add_argument("--config_s", type=str, default=None, help="Short-term model config for two-stage evaluation")
    parser.add_argument("--config_l", type=str, default=None, help="Long-term model config for two-stage evaluation")
    parser.add_argument("--two_stage_n1", type=int, default=None, help="Number of future frames generated by model_s before model_l")
    
    args = parser.parse_args()

    if args.eval_mode == "single":
        if args.ckpt is None:
            parser.error("--ckpt is required when --eval_mode single")
        args.config = resolve_config_path(args.ckpt, args.config)
    else:
        missing = []
        if args.model_s is None:
            missing.append("--model_s")
        if args.model_l is None:
            missing.append("--model_l")
        if args.two_stage_n1 is None:
            missing.append("--two_stage_n1")
        if missing:
            parser.error(f"{', '.join(missing)} required when --eval_mode two_stage")
        args.config_s = resolve_config_path(args.model_s, args.config_s)
        args.config_l = resolve_config_path(args.model_l, args.config_l)
        if args.config is None:
            args.config = args.config_l

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
    if args.eval_mode == "single":
        print("🧠 Initializing Model...")
        model = build_model_from_checkpoint(config, args.ckpt, device)
        result_label = args.ckpt
    else:
        print("🧠 Initializing Two-Stage Models...")
        if not os.path.exists(args.config_s):
            raise FileNotFoundError(f"Config file not found: {args.config_s}")
        if not os.path.exists(args.config_l):
            raise FileNotFoundError(f"Config file not found: {args.config_l}")
        config_s = load_config(args.config_s)
        config_l = load_config(args.config_l)
        if config["data"]["input_n"] != config_l["data"]["input_n"]:
            raise ValueError(
                "Two-stage eval config input_n must match config_l input_n "
                f"({config['data']['input_n']} != {config_l['data']['input_n']})."
            )
        model_s = build_model_from_checkpoint(config_s, args.model_s, device)
        model_l = build_model_from_checkpoint(config_l, args.model_l, device)
        model = TwoStageEvaluatorModel(
            model_s=model_s,
            model_l=model_l,
            config_l=config_l,
            n1=args.two_stage_n1,
            device=device,
        ).to(device)
        model.eval()
        result_label = f"model_s={args.model_s}, model_l={args.model_l}, n1={args.two_stage_n1}"
    
    # Load Validation Dataset (Split 2 for Test partition)
    print("📂 Loading Dataset (Split 2)...")

    val_dataset, _   = build_dataset(config, split=2)
    

    
    indexed_val_dataset = IndexedDataset(val_dataset)
    dataloader = DataLoader(
        indexed_val_dataset,
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4
    )
    
    evaluator = MetricsEvaluator(config=config, device=device)
    
    # Run Evaluation
    print(f"🚀 Starting Evaluation (mode={args.eval_mode}, n_sample={args.nsample})...")
    try:
        use_multigt = args.enable_multigt and isinstance(val_dataset, H36MUnified)
        if use_multigt:
            threshold = args.multimodal_threshold
            if threshold is None:
                threshold = config['data'].get('multimodal_threshold', 0.5)
            multimodal_traj = build_h36m_multimodal_gt(val_dataset, threshold)
            metrics = evaluate_with_multimodal_gt(
                model=model,
                dataloader=dataloader,
                config=config,
                device=device,
                nsample=args.nsample,
                multimodal_traj=multimodal_traj,
            )
        else:
            metrics = evaluator.evaluate(
                model=model,
                dataloader=dataloader,
                nsample=args.nsample
            )
        
        print("\n" + "="*40)
        print(f"{'METRIC':<20} | {'VALUE':<15}")
        print("-" * 40)
        for metric, value in metrics.items():
            if value is None or (isinstance(value, float) and np.isnan(value)):
                print(f"{metric:<20} | NaN")
            else:
                print(f"{metric:<20} | {value:.6f}")
        print("="*40 + "\n")
        print(f"Results for: {result_label}")
        
    except KeyboardInterrupt:
        print("\n❌ Evaluation interrupted.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n❌ Evaluation failed: {e}")

if __name__ == "__main__":
    main()
