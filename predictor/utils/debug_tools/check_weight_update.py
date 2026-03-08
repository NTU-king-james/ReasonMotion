import torch
import torch.nn as nn
import os
import argparse
import sys
import yaml
from model import ModelMain

def get_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def load_model(config, device, checkpoint_path):
    model = ModelMain(config, device, target_dim=3 * 24) # target_dim hardcoded based on context
    
    # Load checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint, strict=False)
    else:
        print(f"Checkpoint not found: {checkpoint_path}")
        return None
        
    model.to(device)
    model.eval()
    return model

def compare_weights(ckpt1, ckpt2, config_path):
    device = torch.device('cpu') # Use CPU for comparison to avoid OOM
    config = get_config(config_path)
    
    print("Loading Model 1...")
    model1 = load_model(config, device, ckpt1)
    if model1 is None: return

    print("Loading Model 2...")
    model2 = load_model(config, device, ckpt2)
    if model2 is None: return

    print("\n--- Comparing Weights ---")
    total_diff = 0.0
    total_params = 0
    max_diff = 0.0
    
    for (name1, p1), (name2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
        if name1 != name2:
            print(f"Mismatch in parameter names: {name1} vs {name2}")
            break
            
        diff =  torch.norm(p1 - p2).item()
        if diff > max_diff:
            max_diff = diff
            
        total_diff += diff
        total_params += p1.numel()
        
        if diff > 1e-4:
             print(f"Update detected in {name1}: L2 Diff = {diff:.6f}")
             
    print(f"\nTotal L2 Difference across all layers: {total_diff:.6f}")
    print(f"Max Layer Difference: {max_diff:.6f}")
    
    if total_diff == 0:
        print("\nWARNING: The weights are identical! The model did not update.")
    else:
        print(f"\nSUCCESS: The weights have changed. Total change magnitude: {total_diff}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, required=True, help="Path to base/pretrained checkpoint")
    parser.add_argument("--new", type=str, required=True, help="Path to new trained checkpoint")
    parser.add_argument("--config", type=str, default="configs/train_grpo.yaml", help="Path to config file")
    
    args = parser.parse_args()
    
    compare_weights(args.base, args.new, args.config)
