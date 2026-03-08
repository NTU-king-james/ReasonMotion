
path = "/home/allen/Diffusion/ReasonMotion_SFT_GRPO_Trajectory/utils/rl_visualizer.py"
with open(path, 'r') as f:
    lines = f.readlines()

new_lines = []
skip = False
found = False
for line in lines:
    if "def _run_training_exploration" in line:
        skip = True
        found = True
        # Insert new code
        new_lines.append('    def _run_training_exploration(self, epoch, model, reward_model, text_encoder, current_std, num_variants):\n')
        new_lines.append('        """\n')
        new_lines.append('        Updated Training Exploration:\n')
        new_lines.append('        Now uses Trajectory-based sampling (same as training).\n')
        new_lines.append('        - Shared Initial Noise x_T\n')
        new_lines.append('        - G Variants generated via stochastic branching\n')
        new_lines.append('        """\n')
        new_lines.append('        torch.cuda.empty_cache()\n')
        new_lines.append('        torch.manual_seed(42)\n')
        new_lines.append('        \n')
        new_lines.append('        feed, text_cond, gt_pose = self._prepare_common_inputs(text_encoder)\n')
        new_lines.append('        \n')
        new_lines.append('        # 1. Generate Variants (G paths from same x_T)\n')
        new_lines.append('        # final_samples: (1, G, K, L)\n')
        new_lines.append('        final_samples, _, _ = model.sample_trajectory(text_cond[0], feed, G=num_variants)\n')
        new_lines.append('        variants = final_samples[0] # (G, K, L)\n')
        new_lines.append('        \n')
        new_lines.append('        # 2. Render\n')
        new_lines.append('        self._rank_and_render(epoch, "training_trajectory", variants, reward_model, num_variants, pure_pred=None)\n')
        new_lines.append('\n')
    
    if "def _run_inference_stability" in line:
        skip = False
    
    if not skip:
        new_lines.append(line)

if not found:
    print("Error: Target method not found!")
    exit(1)

with open(path, 'w') as f:
    f.writelines(new_lines)

print("Successfully updated rl_visualizer.py")
