#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils/motionfix_loader.py

Function:
1. Read MotionLab's MotionFix static dataset (source/target features).
2. On-the-fly recover 22-joint HML features (263 dim) to XYZ.
3. Convert 22-joint XYZ to ReasonMotion compatible 24-joint XYZ (72 dim).
4. Simulate FineFS Dataloader output format.
"""
import os
import sys
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from os.path import join as pjoin

# --- Key: Import MotionLab's tools ---
# Ensure RFMOTION_ROOT is in sys.path
RFMOTION_ROOT = "/home/kingjames23/MotionLab" # Modify according to your path
if RFMOTION_ROOT not in sys.path:
    sys.path.append(RFMOTION_ROOT)
    
try:
    import rfmotion.data.humanml.scripts.motion_process as mp
    from rfmotion.data.humanml.utils.paramUtil import t2m_kinematic_chain
    from rfmotion.data.humanml.common.skeleton import Skeleton
    # Inject necessary global variables of MotionLab (from v2.7 script)
    mp.kinematic_chain = t2m_kinematic_chain
    print("✅ Successfully imported MotionLab recover_from_ric function.")
except Exception as e:
    print(f"❌ Error occurred while importing MotionLab tools: {e}")
    print(f"Please check if RFMOTION_ROOT ( {RFMOTION_ROOT} ) path is correct.")
    sys.exit(1)


class MotionFixAsReasonMotion(Dataset):
    def __init__(self, data_dir, split, seq_len, device='cpu', **kwargs):
        """
        data_dir: Should point to MotionLab's '.../datasets/motionfix' directory
        split: 0 (train), 1 (val), 2 (test)
        seq_len: Required sequence length for ReasonMotion (e.g., 120)
        """
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.device = torch.device(device)
        
        # 1. Load Mean / Std
        self.mean = torch.from_numpy(np.load(pjoin(self.data_dir, 'Mean.npy'))).float().to(self.device)
        self.std = torch.from_numpy(np.load(pjoin(self.data_dir, 'Std.npy'))).float().to(self.device)
        
        # 2. Determine split file
        if split == 0:
            split_file = pjoin(self.data_dir, 'train_motionfix.txt')
        elif split == 1:
            split_file = pjoin(self.data_dir, 'val_motionfix.txt')
        else:
            split_file = pjoin(self.data_dir, 'test_motionfix.txt')
            
        # 3. Read pair_id list
        with open(split_file, 'r') as f:
            self.pair_ids = [line.strip() for line in f.readlines() if line.strip()]
            
        print(f"✅ [MotionFixLoader] Successfully loaded {split_file}, total {len(self.pair_ids)} records.")
        
        # Bind recovery function
        self.recover_func = mp.recover_from_ric

    def __len__(self):
        return len(self.pair_ids)

    def __getitem__(self, idx):
        pair_id = self.pair_ids[idx]
        
        # 1. Load MotionFix data
        try:
            data = np.load(pjoin(self.data_dir, 'new_joint_vecs', f"{pair_id}.npy"), allow_pickle=True).item()
            with open(pjoin(self.data_dir, 'texts', f"{pair_id}.txt"), 'r') as f:
                text = f.read().strip()
                
            source_feat = torch.from_numpy(data['source']).float().to(self.device)
            target_feat = torch.from_numpy(data['target']).float().to(self.device)
        except Exception as e:
            print(f"❌ Error loading {pair_id}: {e}")
            return self.__getitem__((idx + 1) % len(self)) # Skip when error occurs, read next batch

        # 2. (On-the-fly) Feature -> 22 joints XYZ
        denorm_source = source_feat * self.std + self.mean
        denorm_target = target_feat * self.std + self.mean
        
        # recover_func needs (batch, seq, dim) -> (batch, seq, joints, 3)
        source_xyz22 = self.recover_func(denorm_source.unsqueeze(0), 22).squeeze(0) # (T, 22, 3)
        target_xyz22 = self.recover_func(denorm_target.unsqueeze(0), 22).squeeze(0) # (T, 22, 3)
        
        # 3. (On-the-fly) 22 joints -> 24 joints (Y axis inversion + 0 padding)
        T = source_xyz22.shape[0]
        source_xyz24 = torch.zeros(T, 24, 3, dtype=torch.float32, device=self.device)
        target_xyz24 = torch.zeros(T, 24, 3, dtype=torch.float32, device=self.device)

        # Invert Y axis (restore flipY of v2.7 script)
        source_xyz22[..., 1] *= -1.0
        target_xyz22[..., 1] *= -1.0
        
        # Fill first 22 joints
        source_xyz24[:, :22, :] = source_xyz22
        target_xyz24[:, :22, :] = target_xyz22
        
        # 4. Flatten to (T, 72)
        source_flat = source_xyz24.reshape(T, -1) # (T, 72)
        target_flat = target_xyz24.reshape(T, -1) # (T, 72)

        # 5. Process sequence length (Padding / Truncating)
        # ⚠️ Note: the logic here **MUST** be identical to your utils/finefs.py
        # Here we implement a basic version first
        current_len = T
        if current_len < self.seq_len:
            # --- Padding ---
            padding_len = self.seq_len - current_len
            source_padded = torch.cat([source_flat, torch.zeros(padding_len, 72, device=self.device)], dim=0)
            target_padded = torch.cat([target_flat, torch.zeros(padding_len, 72, device=self.device)], dim=0)
            mask = torch.cat([torch.ones(current_len), torch.zeros(padding_len)], dim=0).bool()
        else:
            # --- Truncating ---
            start_idx = 0 # Simply truncate from start
            source_padded = source_flat[start_idx : start_idx + self.seq_len]
            target_padded = target_flat[start_idx : start_idx + self.seq_len]
            mask = torch.ones(self.seq_len).bool()
            current_len = self.seq_len

        # 6. Pack return values (format must be identical to FineFS Dataloader)
        # Assume FineFS returns dict
        return {
            'source_motion': source_padded, # (seq_len, 72)
            'target_motion': target_padded, # (seq_len, 72)
            'text': text,
            'mask': mask, # (seq_len,)
            'motion_len': current_len
        }