#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
/home/allen/Diffusion/DePOSit_Skating_Editor_BaseModel/utils/motionfix_statistics.py

Function:
1. Read MotionFix dataset (train/val .txt files).
2. Iterate through all corresponding text instructions in the 'texts/' directory.
3. Count the total number of instructions, uniqueness, common instructions, common words, vocabulary size, and length distribution.
4. Save the statistical results to `motionfix_text_statistics.txt` in the root directory.
"""

import os
import argparse
import numpy as np
from collections import Counter
from tqdm import tqdm
import re

def main():
    parser = argparse.ArgumentParser(description="Count text instructions in MotionFix dataset")
    parser.add_argument("--motionfix_data_dir", type=str, default='/home/kingjames23/MotionLab/datasets/all', 
                        help="Path to MotionFix dataset (e.g., /home/kingjames23/MotionLab/datasets/all)")
    parser.add_argument("--top_n", type=int, default=30,
                        help="Number of most common instructions/words to display")
    args = parser.parse_args()

    texts_dir = os.path.join(args.motionfix_data_dir, 'texts')
    txt_files_to_check = [
        os.path.join(args.motionfix_data_dir, 'train_motionfix.txt'),
        os.path.join(args.motionfix_data_dir, 'val_motionfix.txt')
    ]
    
    if not os.path.isdir(texts_dir):
        print(f"❌ Cannot find 'texts' directory at: {texts_dir}")
        print("Please ensure --motionfix_data_dir points to the correct path.")
        return

    all_captions = []
    
    for txt_file in txt_files_to_check:
        if not os.path.exists(txt_file):
            print(f"⚠️ Cannot find {txt_file}, skipping...")
            continue
            
        print(f"▶ Analyzing {txt_file} ...")
        with open(txt_file, 'r') as f:
            pair_ids = [line.strip() for line in f.readlines() if line.strip()]
        
        for pair_id in tqdm(pair_ids, desc=f"Reading {os.path.basename(txt_file)}"):
            caption_path = os.path.join(texts_dir, f"{pair_id}.txt")
            try:
                with open(caption_path, 'r', encoding='utf-8') as f_cap:
                    caption = f_cap.read().strip()
                    if caption:
                        all_captions.append(caption)
            except FileNotFoundError:
                print(f"\nWarning: Cannot find {caption_path}, skipping pair_id: {pair_id}")
            except Exception as e:
                print(f"\nWarning: Failed to read {caption_path}: {e}")

    if not all_captions:
        print("❌ Error: No valid text instructions found.")
        return

    print("\n✅ Data reading completed, computing statistics...")

    # --- 1. Overall Statistics ---
    total_samples = len(all_captions)
    caption_counts = Counter(all_captions)
    total_unique_captions = len(caption_counts)

    # --- 2. Word Statistics ---
    # Use re.findall to find all words (including 'don't') and convert to lowercase
    all_words = []
    for caption in all_captions:
        words = re.findall(r"\b[\w']+\b", caption.lower())
        all_words.extend(words)
        
    word_counts = Counter(all_words)
    vocabulary_size = len(word_counts)

    # --- 3. Length Statistics ---
    caption_lengths = [len(re.findall(r"\b[\w']+\b", caption)) for caption in all_captions]
    avg_len = np.mean(caption_lengths)
    min_len = np.min(caption_lengths)
    max_len = np.max(caption_lengths)

    # --- 4. Write Report ---
    output_path = os.path.join(args.motionfix_data_dir, "motionfix_text_statistics.txt")
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("==============================================\n")
            f.write("📊 MotionFix Text Instructions Statistical Report\n")
            f.write("==============================================\n\n")
            
            f.write("--- 1. Overall Overview ---\n")
            f.write(f"Total Instructions (Train + Val): {total_samples}\n")
            f.write(f"Unique Instructions:   {total_unique_captions}\n")
            f.write(f"Unique Instructions Ratio:          {total_unique_captions / total_samples * 100:.2f}%\n")
            f.write("\n")

            f.write("--- 2. Length Distribution (by Word Count) ---\n")
            f.write(f"Shortest Instruction Length: {min_len} words\n")
            f.write(f"Longest Instruction Length: {max_len} words\n")
            f.write(f"Average Instruction Length: {avg_len:.2f} words\n")
            f.write("\n")

            f.write("--- 3. Vocabulary Statistics ---\n")
            f.write(f"Total Words:   {len(all_words)}\n")
            f.write(f"Vocabulary Size: {vocabulary_size}\n")
            f.write("\n")

            f.write(f"--- 4. Top {args.top_n} Common Instructions (Full Captions) ---\n")
            for i, (caption, count) in enumerate(caption_counts.most_common(args.top_n), 1):
                f.write(f"{i:2d}. (Appeared {count:4d} times) \"{caption}\"\n")
            f.write("\n")

            f.write(f"--- 5. Top {args.top_n} Common Words ---\n")
            for i, (word, count) in enumerate(word_counts.most_common(args.top_n), 1):
                f.write(f"{i:2d}. (Appeared {count:5d} times) \"{word}\"\n")
            f.write("\n")

        print(f"✅ Statistical report saved to: {output_path}")

    except Exception as e:
        print(f"❌ Failed to save report: {e}")


if __name__ == "__main__":
    main()