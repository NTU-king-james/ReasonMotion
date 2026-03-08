
import pandas as pd
import numpy as np
import os

LOG_PATH = "/home/allen/Diffusion/ReasonMotion_SFT_GRPO_Trajectory/runs/0129_kl_clip_and_update/loss/batch_metrics.csv"

def analyze_log():
    if not os.path.exists(LOG_PATH):
        print(f"File not found: {LOG_PATH}")
        return

    try:
        df = pd.read_csv(LOG_PATH)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    print(f"Total Batches Recorded: {len(df)}")
    
    # Extract Batch Number from Batch_ID (e.g., E1_B50 -> 50)
    # Assuming standard format E{epoch}_B{batch}
    # But user mentioned "training for over a lap" (epoch?), let's just use index if parsing fails, but parsing is better
    try:
        df['Batch_Num'] = df['Batch_ID'].apply(lambda x: int(x.split('_B')[1]))
    except:
        df['Batch_Num'] = np.arange(len(df))

    # Calculate Moving Averages (Window=10 for smoothness)
    window = 5
    print(f"\n--- Statistics (Last {window} records vs First {window} records) ---")
    
    cols_to_analyze = ['Policy_Loss', 'GRPO_Loss', 'KL_Div', 'R_GT', 'R_Score']
    
    for col in cols_to_analyze:
        if col not in df.columns:
            continue
            
        start_avg = df[col].head(window).mean()
        end_avg = df[col].tail(window).mean()
        delta = end_avg - start_avg
        print(f"{col}: Start={start_avg:.4f} -> End={end_avg:.4f} | Delta={delta:.4f}")

    # Check for correlation (Reward Hacking Check)
    # Does R_Score increase while R_GT decreases?
    print(f"\n--- Correlation Analysis ---")
    corr = df[['R_GT', 'R_Score', 'KL_Div']].corr()
    print(corr)

    # Detect KL Explosion
    max_kl = df['KL_Div'].max()
    print(f"\nMax KL_Div: {max_kl:.4f}")
    if max_kl > 50:
        print("⚠️  WARNING: KL Divergence is dangerously high!")

    # Sample some middle and end rows
    print("\n--- Raw Data Samples (Start, Middle, End) ---")
    print(df.iloc[[0, len(df)//2, -1]].to_string())

if __name__ == "__main__":
    analyze_log()
