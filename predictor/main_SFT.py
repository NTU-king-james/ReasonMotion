# main.py
import os, datetime, torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
from model.model import ModelMain
from motion_data.finefs import FineFS
from motion_data.h36m  import H36M           # ← 新增
from utils.text_encoder import TextEncoder
from utils.config_util_sft import get_config, save_config, load_resume_config, parse_args
from train_SFT import train

visible = os.getenv("CUDA_VISIBLE_DEVICES", "all")   # 沒設就回傳 "all"
print(f"Visible GPUs (CUDA_VISIBLE_DEVICES): {visible}")

# ② 選第一張卡作為主 device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- GPU Info ---
print("PyTorch", torch.__version__)
if torch.cuda.is_available():
    print("CUDA", torch.version.cuda,
          "| Current device:", torch.cuda.current_device(),
          "| Name:", torch.cuda.get_device_name(torch.cuda.current_device()))
# ---------------------- Helper ------------------------ #
def build_dataset(cfg, split):
    """依 cfg['data']['dataset'] 回傳對應 Dataset 物件"""
    ds_name   = cfg['data'].get('dataset', 'finefs').lower()
    common_kw = dict(
        input_n    = cfg['data']['input_n'],
        output_n   = cfg['data']['output_n'],
        skip_rate  = cfg['data'].get('skip_rate', 1),
        split      = split,
        max_len    = cfg['data'].get('max_len'),
    )
    if split == 1:
        common_kw['skip_rate'] = 10  # 驗證集只取 1/10，快速驗證
    if ds_name == 'finefs':
        return FineFS(
            data_dir = cfg['data']['data_dir'],
            mode     = cfg['data'].get('mode', 'full_name'),
            **common_kw
        ), 24 * 3
    elif ds_name == 'h36m':
        joints = 17
        return H36M(
            data_dir  = cfg['data']['data_dir_h36m'],
            joints    = joints,
            **common_kw
        ), joints * 3
    else:
        raise ValueError(f"Unknown dataset: {ds_name}")

# ---------------------- Main -------------------------- #
if __name__ == '__main__':
    args   = parse_args()

    # ---------- 讀取 / 產生 config ---------- #
    if args.resume_dir:                    # Resume
        output_dir = args.resume_dir
        config     = load_resume_config(output_dir)
        load_state = True
        print(f"[Resume] {output_dir}")
    else:                                  # New training
        config     = get_config()
        now        = datetime.datetime.now().strftime("%m%d-%H%M")
        ds_tag     = config['data'].get('dataset', 'finefs')
        input_n    = config['data']['input_n']
        output_n   = config['data']['output_n']
        output_dir = config['data'].get('output_dir', 'default')
        if output_dir == 'default' or not output_dir:
            output_dir = f"runs/{ds_tag}_{input_n}_{output_n}_{now}"
        
        # os.makedirs(output_dir, exist_ok=True)  <-- Moved down
        # save_config(config, output_dir)         <-- Moved down
        load_state = False
        print(f"[New]  Folder: {output_dir}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---------- Dataset ---------- #
    train_ds, target_dim = build_dataset(config, split=0)
    valid_ds, _          = build_dataset(config, split=1)
    print(f"Train / Val = {len(train_ds)} / {len(valid_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=config['train']['batch_size'],
        shuffle=True, num_workers=0
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=config['train']['batch_size'],
        shuffle=True, num_workers=0
    )

    # 0. 根據 Config 自動調整 textemb 維度
    txt_cfg = config.get("text_injection", {})
    if txt_cfg.get("use_clip_encoder", False):
        print("[Auto-Config] Detected CLIP usage, updating textemb to 512.")
        config["model"]["textemb"] = 512
    else:
         # 確保 SBERT 至少是 384
         config["model"]["textemb"] = 384

    # ---------- Save Config Late (Fix) ---------- #
    if not args.resume_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_config(config, output_dir)
        print(f"[Config] Saved to {output_dir}/config.yaml")

    # ---------- Model & TextEncoder ---------- #
    model        = ModelMain(config, device, target_dim=target_dim).to(device)
    if torch.cuda.device_count() > 1:            # ⭐ 自動多卡
        print(f"🖥️  Using {torch.cuda.device_count()} GPUs: DataParallel")
        model = nn.DataParallel(model)           # 最省事的包法
    
    # 提取文字注入配置
    txt_cfg = config.get("text_injection", {})
    print("\n[Config] Text Injection Settings loaded:")
    for k, v in txt_cfg.items():
        print(f"  • {k}: {v}")
    print("")

    text_encoder = TextEncoder(
        device=device,
    ).to(device)

    # ---------- TensorBoard ---------- #
    tb_writer = SummaryWriter(os.path.join(output_dir, "logs"))

    # ---------- Train ---------- #
    train(
        model        = model,
        config       = config['train'],
        train_loader = train_loader,
        valid_loader = valid_loader,
        foldername   = output_dir,
        load_state   = load_state,
        text_encoder = text_encoder,
        tb_writer    = tb_writer,
        target_dim   = target_dim,
        input_n      = config['data']['input_n'],
    )
