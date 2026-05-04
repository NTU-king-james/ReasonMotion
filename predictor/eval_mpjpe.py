"""
Short-term MPJPE evaluation for H36M.

Protocol:
- Evaluate one prediction per sample (deterministic by default).
- Report MPJPE (mm) at each timestamp independently.
- Default timestamps: 80/160/320/400 ms.

Example:
CUDA_VISIBLE_DEVICES=0 python eval_mpjpe.py \
    --ckpt /home/kingjames23/ReasonMotion-project/ReasonMotion/predictor/runs/0503_moe_rl_h36m_smooth_guard/checkpoints/checkpoint_ep1_batch500.pth \
    --config /home/kingjames23/ReasonMotion-project/ReasonMotion/predictor/runs/0503_moe_rl_h36m_smooth_guard/config.yaml\
    --batch_size 128
"""

import argparse
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model.model import ModelMain
from motion_data.finefs import FineFS
from motion_data.h36m_unified import H36MUnified
from utils.text_encoder import TextEncoder


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_config_path(ckpt_path: str, config_path: Optional[str]) -> str:
    if config_path is not None:
        return config_path

    derived = os.path.abspath(os.path.join(os.path.dirname(ckpt_path), "..", "config.yaml"))
    if os.path.exists(derived):
        print(f"Found config file at: {derived}")
        return derived

    fallback = "configs/train_grpo.yaml"
    print(f"Config file not found at {derived}, using default: {fallback}")
    return fallback


def parse_timestamps(timestamp_str: str) -> List[int]:
    values = [int(x.strip()) for x in timestamp_str.split(",") if x.strip()]
    if not values:
        raise ValueError("No timestamps provided.")
    for v in values:
        if v <= 0:
            raise ValueError(f"Timestamp must be positive, got {v}.")
    return values


def timestamps_to_future_indices(timestamps_ms: List[int], fps: float) -> Tuple[List[int], List[int]]:
    frame_steps = []
    frame_indices = []
    for t_ms in timestamps_ms:
        step = int(round(t_ms * fps / 1000.0))
        if step < 1:
            raise ValueError(
                f"Timestamp {t_ms} ms maps to invalid future step {step} at fps={fps}."
            )
        frame_steps.append(step)
        frame_indices.append(step - 1)
    return frame_steps, frame_indices


def build_dataset(cfg: Dict, split: int):
    ds_name = cfg["data"].get("dataset", "h36m").lower()
    print(f"Building dataset: {ds_name} (split={split})")

    common_kw = dict(
        input_n=cfg["data"]["input_n"],
        output_n=cfg["data"]["output_n"],
        skip_rate=cfg["data"].get("skip_rate", 1),
        split=split,
        max_len=cfg["data"].get("max_len"),
    )

    if ds_name == "finefs":
        dataset = FineFS(
            data_dir=cfg["data"]["data_dir"],
            mode=cfg["data"].get("mode", "full_name"),
            **common_kw,
        )
        return dataset, 24 * 3

    if ds_name == "h36m":
        joints = cfg["data"].get("joints", 17)
        dataset = H36MUnified(
            data_dir=cfg["data"]["data_dir"],
            joints=joints,
            downsample=cfg["data"].get("downsample", 1),
            no_overlap=cfg["data"].get("no_overlap", False),
            protocol=cfg["data"].get("h36m_protocol", "predictor"),
            miss_type=cfg["data"].get("miss_type", "no_miss"),
            miss_rate=cfg["data"].get("miss_rate", 0.2),
            all_data=cfg["data"].get("all_data", True),
            data_ratio=cfg["data"].get("data_ratio", 1.0),
            pad_short_sequences=cfg["data"].get("pad_short_sequences", False),
            **common_kw,
        )
        return dataset, joints * 3

    raise ValueError(f"Unknown dataset: {ds_name}")


def load_model(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> None:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict):
        if "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        elif "model_state_dict" in state and isinstance(state["model_state_dict"], dict):
            state = state["model_state_dict"]
        elif "model" in state and isinstance(state["model"], dict):
            state = state["model"]

    if not isinstance(state, dict):
        raise ValueError("Checkpoint format not supported: expected a state dict.")

    state = {k.replace("module.", ""): v for k, v in state.items()}

    try:
        model.load_state_dict(state)
        print("Checkpoint loaded with strict=True")
    except Exception:
        model.load_state_dict(state, strict=False)
        print("Checkpoint loaded with strict=False")


@torch.no_grad()
def evaluate_mpjpe_by_timestamps(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    input_n: int,
    timestamps_ms: List[int],
    eval_indices: List[int],
    deterministic: bool,
) -> Tuple[Dict[int, float], int, int]:
    model.eval()
    text_encoder = TextEncoder(device=str(device))

    mpjpe_sum = {t_ms: 0.0 for t_ms in timestamps_ms}
    n_samples = 0
    skipped_batches = 0

    pbar = tqdm(dataloader, desc="Evaluating MPJPE")
    for batch in pbar:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        t_emb = text_encoder(batch.get("motion_name"))

        # Keep noise generation inside ModelMain.impute to avoid external double injection.
        # sample=False disables per-step stochastic noise in the reverse chain.
        noisy_data = None
        sample_flag = not deterministic

        try:
            samples, gt = model.evaluate(
                batch,
                n_samples=1,
                text_embedding=t_emb,
                noisy_data=noisy_data,
                sample=sample_flag,
            )[:2]
        except Exception:
            skipped_batches += 1
            continue

        if gt.dim() == 3:
            gt = gt.unsqueeze(1)

        pred_future = samples[:, 0, :, input_n:]  # (B, K, T_f)
        gt_future = gt[:, 0, :, input_n:]         # (B, K, T_f)

        if pred_future.shape[-1] == 0:
            skipped_batches += 1
            continue

        max_idx = max(eval_indices)
        if max_idx >= pred_future.shape[-1]:
            raise ValueError(
                "Requested timestamp exceeds predicted future length. "
                f"max required index={max_idx}, available future length={pred_future.shape[-1]}."
            )

        bsz, kdim, tf = pred_future.shape
        if kdim % 3 != 0:
            raise ValueError(f"Invalid feature dimension K={kdim}, expected multiple of 3.")
        njoints = kdim // 3

        pred_future = pred_future.permute(0, 2, 1).reshape(bsz, tf, njoints, 3)
        gt_future = gt_future.permute(0, 2, 1).reshape(bsz, tf, njoints, 3)

        for t_ms, idx in zip(timestamps_ms, eval_indices):
            mpjpe_each = torch.norm(pred_future[:, idx] - gt_future[:, idx], dim=-1).mean(dim=-1)
            mpjpe_sum[t_ms] += float((mpjpe_each * 1000.0).sum().item())

        n_samples += bsz
        postfix = {f"{t}ms": f"{(mpjpe_sum[t] / max(1, n_samples)):.3f}" for t in timestamps_ms}
        pbar.set_postfix(postfix)

    if n_samples == 0:
        raise RuntimeError("No valid samples were evaluated. Check dataset and model outputs.")

    avg = {t_ms: mpjpe_sum[t_ms] / n_samples for t_ms in timestamps_ms}
    return avg, n_samples, skipped_batches


def infer_fps(config: Dict, fps_override: Optional[float]) -> float:
    if fps_override is not None:
        return float(fps_override)

    cfg_fps = config["data"].get("fps")
    if cfg_fps is not None:
        return float(cfg_fps)

    dataset_name = config["data"].get("dataset", "").lower()
    if dataset_name == "h36m":
        downsample = config["data"].get("downsample", 2)
        return 50.0 / float(max(1, downsample))

    return 30.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate short-term MPJPE at specific timestamps (H36M protocol)."
    )
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--batch_size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--split", type=int, default=2, help="Dataset split (default: 2 for test)")
    parser.add_argument("--timestamps", type=str, default="80,160,320,400", help="Comma-separated ms")
    parser.add_argument("--fps", type=float, default=None, help="Override FPS for timestamp mapping")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic diffusion sampling (default is deterministic).",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    config_path = resolve_config_path(args.ckpt, args.config)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config = load_config(config_path)

    ds_name = config["data"].get("dataset", "").lower()
    if ds_name != "h36m":
        raise ValueError(
            f"This script follows H36M short-term protocol, but dataset={ds_name!r} in config."
        )

    device_name = args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available, switching to CPU.")
        device_name = "cpu"
    device = torch.device(device_name)

    dataset, inferred_target_dim = build_dataset(config, split=args.split)
    if "target_dim" not in config["model"]:
        config["model"]["target_dim"] = inferred_target_dim

    target_dim = config["model"].get("target_dim", inferred_target_dim)
    model = ModelMain(config=config, device=device, target_dim=target_dim).to(device)
    load_model(model, args.ckpt, device)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    timestamps_ms = parse_timestamps(args.timestamps)
    fps = infer_fps(config, args.fps)
    frame_steps, eval_indices = timestamps_to_future_indices(timestamps_ms, fps)

    output_n = int(config["data"].get("output_n", 0))
    if output_n > 0 and max(eval_indices) >= output_n:
        raise ValueError(
            f"Largest timestamp needs future index {max(eval_indices)}, "
            f"but config output_n={output_n}."
        )

    deterministic = not args.stochastic
    print("=" * 60)
    print(f"Checkpoint         : {args.ckpt}")
    print(f"Config             : {config_path}")
    print(f"Dataset split      : {args.split}")
    print(f"FPS                : {fps:.4f}")
    print(f"Timestamps (ms)    : {timestamps_ms}")
    print(f"Future frame steps : {frame_steps}")
    print(f"Future frame idx   : {eval_indices}")
    print(f"Sampling mode      : {'deterministic' if deterministic else 'stochastic'}")
    print("=" * 60)

    metrics, n_samples, skipped_batches = evaluate_mpjpe_by_timestamps(
        model=model,
        dataloader=dataloader,
        device=device,
        input_n=int(config["data"]["input_n"]),
        timestamps_ms=timestamps_ms,
        eval_indices=eval_indices,
        deterministic=deterministic,
    )

    print("\n" + "=" * 60)
    print(f"{'Timestamp':<12} | {'FrameStep':<10} | {'MPJPE (mm)':<12}")
    print("-" * 60)
    for t_ms, step in zip(timestamps_ms, frame_steps):
        print(f"{t_ms:>4} ms      | {step:>10} | {metrics[t_ms]:>12.6f}")
    print("=" * 60)
    print(f"Evaluated samples: {n_samples}")
    print(f"Skipped batches  : {skipped_batches}")


if __name__ == "__main__":
    main()
