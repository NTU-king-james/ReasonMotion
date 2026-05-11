**結論**
這份 paper 的 predictor model 部分「架構主軸有 follow 程式」，但「實驗與 reward 設定沒有完全 follow 目前這個 H36M run」。特別是 `predictor/runs/h36m_fairscale_rl_with_smooth/config.yaml` 這個設定實際上是 H36M + GT/smooth reward 的 RL fine-tuning，不是 paper 強調的 GOE / sport-specific scoring reward 主導設定。

**目前 Config 重點**
參考 [config.yaml](/home/kingjames23/ReasonMotion-project/ReasonMotion/predictor/runs/h36m_fairscale_rl_with_smooth/config.yaml:1)：

- Dataset: H36M, 17 joints, `input_n=25`, `output_n=100`, `h36m_protocol=deposit`
- Diffusion: 12 layers, 64 channels, 8 heads, 50 diffusion steps, cosine schedule
- MoE: `fairscale_rl`
- Text: `cross_attn`, `textemb=384`
- R3: `use_r3=true`
- RL: batch size 2, group size `G=8`, `sampling_std=0.015`, `lr=1e-5`
- Reward: `w_gt=0.8`, `w_smooth=0.2`, `w_score=0`, `w_rot=0`, `w_righthand=0`

**相同之處**
1. **Text-Conditional Diffusion 有對上 paper。**  
   Paper 說在 ResidualBlock 中用 sentence-transformer text embedding 做 cross-attention。程式中 `TextEncoder` 使用 `all-MiniLM-L6-v2`，輸出 384 維 token embedding：  
   [text_encoder.py](/home/kingjames23/ReasonMotion-project/ReasonMotion/predictor/utils/text_encoder.py:14)  
   Cross-attention 實作在 ResidualBlock：motion query 對 text key/value：  
   [diffusion_util.py](/home/kingjames23/ReasonMotion-project/ReasonMotion/predictor/model/diffusion_util.py:167)

2. **ST-MoE 的 temporal + spatial/feature 兩段式結構有對上。**  
   Paper 說每個 residual block 依序做 Temporal MoE 和 Spatial MoE。程式中每個 ResidualBlock 建立 `time_layer` 與 `feature_layer`，且都是 MoE Transformer：  
   [diffusion_util.py](/home/kingjames23/ReasonMotion-project/ReasonMotion/predictor/model/diffusion_util.py:190)  
   forward 順序也是 time 再 feature：  
   [diffusion_util.py](/home/kingjames23/ReasonMotion-project/ReasonMotion/predictor/model/diffusion_util.py:278)

3. **MoE FFN 取代 Transformer FFN 有對上。**  
   `MoETransformerEncoderLayer` override `_ff_block()`，用 MoE layer 取代原本 FFN：  
   [moe_transformer.py](/home/kingjames23/ReasonMotion-project/ReasonMotion/predictor/model/moe_transformer.py:21)

4. **vis-GRPO 的 group relative advantage、clipped ratio、KL reference 方向有對上。**  
   group reward normalization：  
   [rl_utils.py](/home/kingjames23/ReasonMotion-project/ReasonMotion/predictor/utils/rl_utils.py:675)  
   PPO/GRPO clipped surrogate loss：  
   [model.py](/home/kingjames23/ReasonMotion-project/ReasonMotion/predictor/model/model.py:540)  
   KL against reference policy：  
   [model.py](/home/kingjames23/ReasonMotion-project/ReasonMotion/predictor/model/model.py:549)

5. **R3 routing replay 有對上 paper。**  
   rollout 時收集 routing：  
   [model.py](/home/kingjames23/ReasonMotion-project/ReasonMotion/predictor/model/model.py:360)  
   training pass 注入 routing：  
   [model.py](/home/kingjames23/ReasonMotion-project/ReasonMotion/predictor/model/model.py:524)  
   masked softmax replay 在 FairscaleMoEBlock_RL：  
   [moe_transformer.py](/home/kingjames23/ReasonMotion-project/ReasonMotion/predictor/model/moe_transformer.py:692)

**不同之處**
1. **最大差異：paper 說用 scoring / GOE reward，但目前 H36M config 沒有。**  
   Config 中 `w_score=0.0`，所以 `Scoring Reward Model` 不會啟用：  
   [config.yaml](/home/kingjames23/ReasonMotion-project/ReasonMotion/predictor/runs/h36m_fairscale_rl_with_smooth/config.yaml:60)  
   程式也只有 `w_score > 0` 且有 checkpoint 時才會載入 score model：  
   [rl_utils.py](/home/kingjames23/ReasonMotion-project/ReasonMotion/predictor/utils/rl_utils.py:221)

2. **目前 reward 實際是 GT similarity + smoothness，不是 paper 的完整 reward。**  
   實際 total reward：`w_gt * r_gt + w_smooth * r_smooth + w_score * r_score + w_rot * r_rot`：  
   [rl_utils.py](/home/kingjames23/ReasonMotion-project/ReasonMotion/predictor/utils/rl_utils.py:437)  
   目前 config 只開 `w_gt=0.8`、`w_smooth=0.2`。Paper 提到的 bone consistency reward 沒看到實作。

3. **H36M 的 text condition 不是 paper 寫的 localized edit instruction。**  
   Paper 說 Human3.6M 使用類似 “lower right hand, walking” 的文字。實際 H36M loader 只回傳 action label，例如 `walking`：  
   [h36m_unified.py](/home/kingjames23/ReasonMotion-project/ReasonMotion/predictor/motion_data/h36m_unified.py:263)  
   RL 訓練直接 encode `batch["motion_name"]`：  
   [rl_utils.py](/home/kingjames23/ReasonMotion-project/ReasonMotion/predictor/utils/rl_utils.py:638)

4. **paper 的 Gaussian policy 描述較簡化；程式實際是 trajectory-based diffusion log-prob。**  
   Paper Eq. 7/8 寫成 final output Gaussian。程式真正訓練走完整 denoising chain，對每個 reverse step 算 log-prob：  
   rollout sampling：  
   [model.py](/home/kingjames23/ReasonMotion-project/ReasonMotion/predictor/model/model.py:309)  
   per-step log-prob：  
   [model.py](/home/kingjames23/ReasonMotion-project/ReasonMotion/predictor/model/model.py:533)

5. **paper 說 standard Top-K softmax；目前 config 用 Fairscale/GShard Top-2 routing。**  
   `fairscale_rl` 實際是 top-1 argmax + Gumbel-max 選第二 expert，不是單純 Top-K softmax：  
   [moe_transformer.py](/home/kingjames23/ReasonMotion-project/ReasonMotion/predictor/model/moe_transformer.py:574)  
   而且 expert 數量在 ResidualBlock 裡 hard-code 為 16：  
   [diffusion_util.py](/home/kingjames23/ReasonMotion-project/ReasonMotion/predictor/model/diffusion_util.py:194)

6. **`update_ref_every_batch` 在 config 有寫，但實際被忽略。**  
   Config 設 `update_ref_every_batch=200`：  
   [config.yaml](/home/kingjames23/ReasonMotion-project/ReasonMotion/predictor/runs/h36m_fairscale_rl_with_smooth/config.yaml:54)  
   但 trainer 明確印出 fixed reference mode，KL ref 保持 base checkpoint：  
   [rl_utils.py](/home/kingjames23/ReasonMotion-project/ReasonMotion/predictor/utils/rl_utils.py:627)

**建議 Paper 修正**
如果 paper 要對應這個 H36M run，建議把 predictor 實驗描述改成：

- H36M predictor 使用 action label 作為 text condition，不是 localized edit instruction。
- H36M RL reward 是 `0.8 * exp(-L2 future distance) + 0.2 * exp(-10 acceleration)`，沒有 GOE scoring。
- vis-GRPO 實作是 trajectory-based DDPM reverse-process PPO/GRPO，不是只對 final coordinate Gaussian 做 ratio。
- ST-MoE routing 使用 Fairscale/GShard-style Top-2 with Gumbel second expert。
- Bone consistency reward 若要保留在 paper，需要補 code；否則應從 predictor reward 描述刪除。

========================================
METRIC               | VALUE          
----------------------------------------
AMPJPE               | 64.430721
FMPJPE               | 83.784919
ADE                  | 0.357421
FDE                  | 0.461043
MMADE                | 0.489453
MMFDE                | 0.528610
Diversity            | 2.435871
========================================

Results for: /home/kingjames23/ReasonMotion-project/ReasonMotion/predictor/runs/h36m_fairscale_rl_with_smooth/checkpoints/checkpoint_ep1_batch200.pth