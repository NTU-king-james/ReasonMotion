import os
import csv
import torch
import numpy as np
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.metrics import ampjpe, fmpjpe

def save_training_plot(history, filename_base):
    # Plot 1: Losses
    plt.figure(figsize=(10, 6))
    epochs = history['epoch']
    
    if 'train_loss' in history:
        plt.plot(epochs, history['train_loss'], label='Train Loss', marker='.')
    if 'valid_loss' in history:
        v_mod = [(e, v) for e, v in zip(epochs, history['valid_loss']) if v is not None]
        if v_mod:
             vx, vy = zip(*v_mod)
             plt.plot(vx, vy, label='Valid Loss', marker='o')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{filename_base}_loss.png")
    plt.close()

    # Plot 2: Robustness Ratios
    plt.figure(figsize=(10, 6))
    if 'r_shuf' in history:
         plt.plot(epochs, history['r_shuf'], label='Robustness (Shuffle)', linestyle='-', marker='x')
    if 'r_zero' in history:
         plt.plot(epochs, history['r_zero'], label='Robustness (Zero)', linestyle='-', marker='s')
    
    # Add a baseline at 1.0 (Neutral)
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.3, label='Baseline (1.0)')

    plt.xlabel('Epoch')
    plt.ylabel('Ratio (Error / Main Error)')
    plt.title('Conditioning Robustness (Higher is Better)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{filename_base}_robustness.png")
    plt.close()

    # Plot 3: MPJPE Metrics
    has_ampjpe = 'ampjpe' in history and any(v is not None for v in history['ampjpe'])
    has_fmpjpe = 'fmpjpe' in history and any(v is not None for v in history['fmpjpe'])
    if has_ampjpe or has_fmpjpe:
        plt.figure(figsize=(10, 6))
        if has_ampjpe:
            pts = [(e, v) for e, v in zip(epochs, history['ampjpe']) if v is not None]
            if pts:
                ex, ey = zip(*pts)
                plt.plot(ex, ey, label='A-MPJPE (mm)', marker='o')
        if has_fmpjpe:
            pts = [(e, v) for e, v in zip(epochs, history['fmpjpe']) if v is not None]
            if pts:
                ex, ey = zip(*pts)
                plt.plot(ex, ey, label='F-MPJPE (mm)', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Error (mm)')
        plt.title('MPJPE Metrics')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{filename_base}_mpjpe.png")
        plt.close()


def load_state(model, optimizer, scheduler, foldername, epoch=None):
    if epoch is None:
        ckpt_path = os.path.join(foldername, "params.pth")
    else:
        ckpt_path = os.path.join(foldername, "checkpoints", f"model_ep{epoch}.pth")
    print(f"[Load] {ckpt_path}")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if ckpt_path.endswith(".pth") and "model_ep" in ckpt_path:
        model.load_state_dict(torch.load(ckpt_path))
        return {"epoch": epoch}
    else:
        state = torch.load(ckpt_path)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        return {"epoch": state['epoch']}


def save_state(model, optimizer, scheduler, epoch_no, foldername):
    """儲存模型與訓練狀態"""
    params = {
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch_no
    }
    # 儲存模型參數
    model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(model_state, os.path.join(foldername, "model.pth"))
    # 儲存 optimizer、scheduler 狀態
    torch.save(params, os.path.join(foldername, "params.pth"))


def train(model, config, train_loader, valid_loader=None, valid_epoch_interval=10,
          foldername="", load_state=False, text_encoder=None, tb_writer=None,
          target_dim=72, input_n=0):

    # -------- Optimizer & Scheduler --------
    optimizer = Adam(model.parameters(), lr=config.get('lr', 1e-3), weight_decay=1e-6)
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: min(1.0, step / 2000))
    decay_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=1e-6)

    # -------- Resume --------
    if load_state and os.path.exists(f'{foldername}/params.pth'):
        ckpt = torch.load(f'{foldername}/params.pth')
        optimizer.load_state_dict(ckpt['optimizer'])
        warmup_scheduler.last_epoch = ckpt['epoch']
        decay_scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
    else:
        start_epoch = 0

    # -------- Prepare folders --------
    checkpoints_dir = os.path.join(foldername, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    # -------- Check text encoder --------
    tmp_batch = next(iter(train_loader))
    tok_emb, tok_mask = text_encoder(tmp_batch["motion_name"])
    print("▸ text embed shape:", tok_emb.shape, tok_mask.shape)

    # -------- R3 (Strategy C) Status --------
    inner_model = model.module if isinstance(model, nn.DataParallel) else model
    _use_r3 = getattr(inner_model, 'use_r3', False)
    if _use_r3:
        print("[R3] SFT training — Strategy-C two-pass routing replay enabled")
    if tb_writer:
        tb_writer.add_scalar("config/use_r3", int(_use_r3), 0)

    # -------- Train loop --------
    best_valid = 1e10
    gs = 0
    history = {'epoch': [], 'train_loss': [], 'valid_loss': [], 'r_shuf': [], 'r_zero': [],
               'ampjpe': [], 'fmpjpe': []}

    for epoch in range(start_epoch, config["epochs"]):
        inner_model.use_r3 = _use_r3   # restore R3 for training
        model.train()
        epoch_loss = 0.0

        with tqdm(train_loader, desc=f"Epoch {epoch}", mininterval=5) as it:
            for batch_no, batch in enumerate(it, 1):
                # --- Text Embedding ---
                with torch.no_grad():
                    tok_emb, tok_mask = text_encoder(batch["motion_name"])
                tok_emb = tok_emb.detach()

                # --- CFG Dropout (25%) ---
                if torch.rand(1).item() < 0.05:
                    tok_emb = torch.zeros_like(tok_emb)
                    # tok_mask doesn't matter much if emb is zero, but let's keep it safe
                    # Actually, mask should probably indicate "ignore this" but our attention 
                    # uses padding_mask. Zero vector attention -> zero output usually.

                # --- Main loss ---
                optimizer.zero_grad()
                loss_main = model(batch, text_embedding=(tok_emb, tok_mask)).mean()

                # --- Backward ---
                loss_main.backward()
                pre_clip_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e9)
                post_clip_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                warmup_scheduler.step()
                decay_scheduler.step()

                # --- Logging ---
                epoch_loss += loss_main.item()
                gs += 1
                
                if tb_writer and gs % 100 == 0:
                    tb_writer.add_scalar("opt/lr", optimizer.param_groups[0]['lr'], gs)
                    tb_writer.add_scalar("grad/norm_pre", pre_clip_norm, gs)
                    tb_writer.add_scalar("grad/norm_post", post_clip_norm, gs)
                    tb_writer.add_scalar("loss/main", loss_main.item(), gs)

                it.set_postfix(loss=f"{loss_main.item():.4f}", refresh=False)

        # --- Epoch-end metrics ---
        avg_train = epoch_loss / batch_no
        
        # --- Robustness Check (Average over 20 batches) ---
        inner_model.use_r3 = False   # disable R3 for eval (single-pass suffices)
        model.eval()
        r_shuf_sum = 0.0
        r_zero_sum = 0.0
        r_count = 0
        
        with torch.no_grad():
             loader_to_test = valid_loader if valid_loader else train_loader
             # We want to iterate, but not exhaust if it's an iterator. 
             # Loaders are usually iterators but we can restart them.
             for i, rb_batch in enumerate(loader_to_test):
                 if i >= 20: break # Check 20 batches
                 
                 rb_tok, rb_mask = text_encoder(rb_batch["motion_name"])
                 rb_main = model(rb_batch, text_embedding=(rb_tok, rb_mask)).mean().item()
                 
                 # 1. Shuffle
                 idx = torch.randperm(rb_tok.size(0), device=rb_tok.device)
                 rb_shuf = model(rb_batch, text_embedding=(rb_tok[idx], rb_mask[idx])).mean().item()
                 
                 # 2. Pure Zero (100% drop)
                 rb_zero_tok = torch.zeros_like(rb_tok)
                 rb_zero = model(rb_batch, text_embedding=(rb_zero_tok, rb_mask)).mean().item()
                 
                 # Ratios for this batch
                 r_shuf_sum += (rb_shuf / (rb_main + 1e-9))
                 r_zero_sum += (rb_zero / (rb_main + 1e-9))
                 r_count += 1
             
             ratio_shuf = r_shuf_sum / max(1, r_count)
             ratio_zero = r_zero_sum / max(1, r_count)
             guidance_score = ratio_zero - 1.0
             
             print(f" [Robustness] (avg 20 batches) rShuf: {ratio_shuf:.2f}, rZero: {ratio_zero:.2f}")

        if tb_writer:
            tb_writer.add_scalar("epoch/train_loss", avg_train, epoch)
            tb_writer.add_scalar("robustness/ratio_shuf", ratio_shuf, epoch)
            tb_writer.add_scalar("robustness/ratio_zero", ratio_zero, epoch)
            tb_writer.add_scalar("robustness/guidance_score", guidance_score, epoch)

        # --- Validation ---
        epoch_ampjpe, epoch_fmpjpe = None, None
        if valid_loader and (epoch + 1) % valid_epoch_interval == 0:
            model.eval()
            v_loss = 0.0
            amp_total, fmp_total, n_eval_batches = 0.0, 0.0, 0
            eval_model = model.module if isinstance(model, nn.DataParallel) else model
            with torch.no_grad():
                for v_batch in valid_loader:
                    v_tok, v_mask = text_encoder(v_batch["motion_name"])
                    v_loss += model(v_batch, text_embedding=(v_tok, v_mask)).mean().item()

                    # --- Compute A-MPJPE & F-MPJPE ---
                    try:
                        samples, gt = eval_model.evaluate(
                            v_batch, n_samples=1, text_embedding=(v_tok, v_mask)
                        )[:2]  # (B, N, K, T)
                        if gt.dim() == 3:
                            gt = gt.unsqueeze(1)
                        gt = gt.expand_as(samples)
                        p_part = samples[..., input_n:]
                        g_part = gt[..., input_n:]
                        amp_total += ampjpe(p_part, g_part, target_dim).item()
                        fmp_total += fmpjpe(p_part, g_part, target_dim).item()
                        n_eval_batches += 1
                    except Exception as e:
                        import traceback
                        print(f"  [Warn] MPJPE eval failed on batch: {e}")
                        traceback.print_exc()

            v_loss /= len(valid_loader)
            if n_eval_batches > 0:
                epoch_ampjpe = amp_total / n_eval_batches
                epoch_fmpjpe = fmp_total / n_eval_batches
                print(f"  [Metrics] A-MPJPE: {epoch_ampjpe:.2f} mm | F-MPJPE: {epoch_fmpjpe:.2f} mm")

            if tb_writer:
                tb_writer.add_scalar("epoch/valid_loss", v_loss, epoch)
                if epoch_ampjpe is not None:
                    tb_writer.add_scalar("epoch/A-MPJPE", epoch_ampjpe, epoch)
                    tb_writer.add_scalar("epoch/F-MPJPE", epoch_fmpjpe, epoch)

            if v_loss < best_valid:
                best_valid = v_loss
                save_state(model, optimizer, decay_scheduler, epoch, foldername)
                print(f"✓ new best valid {v_loss:.4f} @ epoch {epoch}")

        # --- Save checkpoint every 1 epochs ---
        ck_name = os.path.join(checkpoints_dir, f"model_ep{epoch+1}.pth")
        torch.save(model.state_dict(), ck_name)
        print(f"checkpoint → {ck_name}")

        # --- Update History & Plot ---
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_train)
        history['r_shuf'].append(ratio_shuf if 'ratio_shuf' in locals() else 0.0)
        history['r_zero'].append(ratio_zero if 'ratio_zero' in locals() else 0.0)
        
        # Valid loss (only add if we ran validation this epoch)
        # Note: v_loss is from inner scope, we strictly need to check if we ran validation
        if valid_loader and (epoch + 1) % valid_epoch_interval == 0:
             history['valid_loss'].append(v_loss)
        else:
             history['valid_loss'].append(None)
        history['ampjpe'].append(epoch_ampjpe)
        history['fmpjpe'].append(epoch_fmpjpe)
             
        # --- Save History to CSV ---
        csv_path = os.path.join(foldername, "training_history.csv")
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, mode='a', newline='') as f:
            headers = ['epoch', 'train_loss', 'valid_loss', 'r_shuf', 'r_zero', 'A-MPJPE', 'F-MPJPE']
            writer = csv.DictWriter(f, fieldnames=headers)
            if not file_exists:
                writer.writeheader()
            
            ran_val = valid_loader and (epoch + 1) % valid_epoch_interval == 0
            row = {
                'epoch': epoch + 1,
                'train_loss': avg_train,
                'valid_loss': v_loss if ran_val else '',
                'r_shuf': ratio_shuf,
                'r_zero': ratio_zero,
                'A-MPJPE': f"{epoch_ampjpe:.2f}" if epoch_ampjpe is not None else '',
                'F-MPJPE': f"{epoch_fmpjpe:.2f}" if epoch_fmpjpe is not None else '',
            }
            writer.writerow(row)

        save_training_plot(history, os.path.join(foldername, "training_metrics"))

        # -------- Save final model --------
        save_state(model, optimizer, decay_scheduler, config["epochs"] - 1, foldername)
