# README.md

SFT (Supervised Fine-Tuning) Training Instructions
- python main_SFT.py --config "your config path"

GRPO (Reinforcement Learning with Proximal Policy Optimization) Training Instructions
- python main_GRPO.py --config "your config path"

Testing Instructions
- python test.py --config "your config path"

Visualization Instructions
- python visualize.py --config "your config path"

for multimodal evaluation
- CUDA_VISIBLE_DEVICES=2 python test.py --ckpt /home/kingjames23/ReasonMotion/predictor/runs/h36m_fairscale_rl/checkpoints/checkpoint_ep1_batch250.pth --nsample 50 --enable_multigt --multimodal_threshold 0.5 --batch_size 128

current best ckpt: /home/allen/Diffusion/ReasonMotion_SFT_GRPO_Trajectory/runs/h36m_rl_gt_smooth/checkpoints/checkpoint_ep1_batch4500.pth