#visualize_infer.py

import os, json, pickle, datetime, argparse, random
from pathlib import Path

import numpy as np
import torch, imageio, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import ModelMain
from utils.finefs import expand_motion_name
from utils.text_encoder import TextEncoder
from utils.config_util import load_config

"""
example usage:
conda activate diff_env

python visualize_infer.py \
  --run_dir "/home/allen/Diffusion/ReasonMotion_SFT_GRPO/runs/12090619_SFT_RL_FineFS" \
  --epoch 23 \
  --res_pk "/home/allen/datasets/FineFS_5s/3_final/test/3Lz/3Lz_0006/new_res.pk" \
  --texts "quadruple, triple, double, single"

    python visualize_infer.py \
    --run_dir "/home/allen/Diffusion/ReasonMotion_SFT_GRPO/runs/0102_SFT_RL_FineFS_vis_01" \
    --epoch 1 \
    --res_pk "/home/allen/datasets/FineFS_5s/3_final/test/3Lz/3Lz_0006/new_res.pk" \
    --texts "quadruple, triple, double, single" --seed 123

    python visualize_infer.py \
    --run_dir "/home/allen/Diffusion/ReasonMotion_SFT_GRPO/runs/0102_SFT_RL_FineFS_vis_01" \
    --res_pk "/home/allen/datasets/FineFS_5s/3_final/valid/4F/4F_0011/new_res.pk" \
    --texts "quadruple, triple, double, single" --seed 123 --epoch 10
"""
# ================ Skeleton & Draw ==================
J = 24
EDGES = [
    (0,1),(1,4),(4,7),(7,10),
    (0,2),(2,5),(5,8),(8,11),
    (0,3),(3,6),(6,9),(9,12),
    (12,15),(12,13),(13,16),(16,18),(18,20),(20,22),
    (12,14),(14,17),(17,19),(19,21),(21,23)
]
def load_model_state(model, ckpt_path):
    print(f"[Load model weights] {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)
def load_model(config, device, checkpoint_path):
    model = ModelMain(config, device, target_dim=J * 3).to(device)
    load_model_state(model, checkpoint_path)
    #model.eval()
    model.train() # [Test] Use train mode to match RL Viz behavior (Dropout on)
    return model

def safe_name(s:str)->str:
    return (s.replace(' ','_').replace('/','_')
             .replace('+','p').replace(',','')
             .replace('=','').replace('.','d'))

def draw(ax, gt, pred, bones, in_n, out_s, idx, text, shift=0.6 ):
    ax.clear()
    floor=0.8
    ax.plot_surface(*np.meshgrid(np.linspace(-floor,floor,2),
                                 np.linspace(-floor,floor,2)),
                    np.full((2,2),-0.6),color='lightgray',alpha=.15)
    g=gt.copy(); g[:,0]-=shift
    ax.scatter(g[:,0],g[:,2],-g[:,1],c='gray',s=20)
    for a,b in bones: ax.plot([g[a,0],g[b,0]], [g[a,2],g[b,2]], [-g[a,1],-g[b,1]], c='gray')
    p=pred.copy(); p[:,0]+=shift
    ax.scatter(p[:,0],p[:,2],-p[:,1],c='blue',s=20)
    for a,b in bones: ax.plot([p[a,0],p[b,0]], [p[a,2],p[b,2]], [-p[a,1],-p[b,1]], c='blue')
    phase = 'Input' if idx<in_n else 'Prediction'
    ax.set_title(f"{text} | Frame {idx}")
    ax.view_init(20,45)
    ax.set_xlim([-1.2,1.2]); ax.set_ylim([-0.6,0.6]); ax.set_zlim([-0.6,0.6])
    ax.set_axis_off()

def seq2combined(gt, preds, mp4, bones, in_n, out_s, texts, fps=25):
    writer=imageio.get_writer(str(mp4),fps=fps,codec='libx264',bitrate='12M')
    fig = plt.figure(figsize=(6*len(texts),6))
    axs = [fig.add_subplot(1, len(texts), i+1, projection='3d', label=text) for i,text in enumerate(texts)]
    #fig.subplots_adjust(wspace=0.05)  # 調小水平間距，預設大約是 0.2
    T = min(pred.shape[0] for pred in preds.values())
    if gt.shape[0]<T:
        pad=np.repeat(gt[-1:],T-gt.shape[0],0)
        gt=np.vstack([gt,pad])

    for t in range(T):
        for ax, text in zip(axs, texts):
            pred = preds[text]
            g = np.stack([gt[t,i::3] for i in range(3)],1)
            p = np.stack([pred[t,i::3] for i in range(3)],1)
            draw(ax, g, p, bones, in_n, out_s, t, text )
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(),np.uint8)
        h,w = fig.canvas.get_width_height()[::-1]
        writer.append_data(frame.reshape(h,w,3))
    writer.close(); plt.close(fig)
    print(f"✅ Saved combined video: {mp4}")


# ===================== Main ========================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="訓練結果資料夾 e.g., /home/user/Diffusion/runs/rotation_30_40_0711-1338")
    ap.add_argument("--epoch", type=int, required=True, help="讀取的 checkpoint epoch")
    ap.add_argument("--res_pk", required=True, help="要測試的 res pk 檔路徑")
    ap.add_argument("--texts", type=str, required=True, help="輸入文字，逗號分隔")
    ap.add_argument("--slidewindow",type=int,default=0)
    ap.add_argument("--seed", type=int, help="固定隨機種子")
    args = ap.parse_args()

    # ------ Seed -------
    if args.seed is None:
        args.seed = random.randint(0, 100000)
    #random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); #torch.cuda.manual_seed_all(args.seed)

    # ------ Config & Model -------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_path = os.path.join(args.run_dir, "config.yaml")
    config = load_config(config_path)
    checkpoint = os.path.join(args.run_dir, "checkpoints", f"model_epoch_{args.epoch}.pth")
    print(f"[Loading model from] {checkpoint}")
    model = load_model(config, device, checkpoint)
    text_encoder = TextEncoder(device=device).to(device)

    input_n = config['data']['input_n']
    output_s = config['data']['output_n']
    total = input_n + output_s

    # ------ Load pose data -------
    with open(args.res_pk, "rb") as f:
        data = pickle.load(f)
    key = 'pred_xyz_24_struct_global' if 'pred_xyz_24_struct_global' in data else 'pred_xyz_24_struct'
    xyz = data[key].astype(np.float32)
    xyz = xyz[args.slidewindow:args.slidewindow+total]
    pose_all = xyz.reshape(-1, J * 3)


    # ------ Inference for each text -------
    text_list = [t.strip() for t in args.texts.split(",")]
    preds = {}
    for text in text_list:
        print(f"▸ Inference text: {text}")
        with torch.no_grad():
            tok_emb, tok_mask = text_encoder([text])
        tok_emb, tok_mask = tok_emb.to(device), tok_mask.to(device)

        pose = torch.tensor(pose_all).unsqueeze(0).to(device)
        mask = torch.zeros_like(pose); mask[:, :input_n] = 1
        tp = torch.arange(pose.shape[1]).unsqueeze(0).float().to(device)

        feed = {"pose": pose, "mask": mask, "timepoints": tp}
        with torch.no_grad():
            p = model.evaluate(feed, 1, text_embedding=(tok_emb, tok_mask))[0].permute(0,1,3,2)[0,0].cpu().numpy()
        preds[text] = p

    # ------ Save video -------
    vis_dir = Path(args.run_dir) / "visualize" / f"epoch{args.epoch}"
    vis_dir.mkdir(parents=True, exist_ok=True)

    res_name = Path(args.res_pk).parent.parent.name
    base_name = f"{res_name}_" + "_VS_".join([safe_name(t) for t in text_list])
    mp4_path = vis_dir / f"{base_name}.mp4"

    # if the file already exists, append a number to the filename
    i = 1
    while mp4_path.exists():
        mp4_path = vis_dir / f"{base_name}_{i}.mp4"
        i += 1

    seq2combined(pose_all, preds, mp4_path, EDGES, input_n, output_s, text_list)