以下是我看過 `train_grpo.py`、`rl_utils.py`、`model.py` 和你那份 CSV 後的判斷。

## 結論

你的問題不只是 reward 權重調不好，而是 **GRPO 實作和 reward 定義都有會導致 smoothness/GT 退化的設計問題**。CSV 也支持這點：`R_Smooth` 從 `0.9439` 掉到 `0.6676`，`R_GT` 從 `0.9791` 掉到 `0.9713`，`R_Total` 也從 `1.0664` 掉到 `0.9773`。這代表目前更新式沒有穩定優化你 log 出來的 reward。

## 主要問題

1. **reward 是 full sequence 算的，input frame 會稀釋錯誤**  
   `compute_gt_reward()` 和 `compute_smoothness_reward()` 都對 input+output 全部 frame 算 reward。10 input / 10 output 時，input 10 frame 最後被 GT 覆蓋，所以 `R_GT` 會被固定正確的 input 拉高，看起來只小跌，但 future 可能已經明顯變差。見 [rl_utils.py](/home/kingjames23/ReasonMotion/predictor/utils/rl_utils.py:354)。

2. **GRPO 用的是 group-relative reward，不保證絕對 reward 上升**  
   目前先做 `(rewards - mean) / std`，再用相對排名更新。這會鼓勵同一組 samples 裡「相對比較好」的樣本，但如果整組樣本分布逐步變壞，平均 smoothness 仍可能一路下滑。見 [rl_utils.py](/home/kingjames23/ReasonMotion/predictor/utils/rl_utils.py:537)。

3. **這份實作不是完整 PPO/GRPO clipping**  
   `epsilon`、`old_model`、`total_log_probs` 幾乎沒有真正參與 clipped ratio objective；實際上比較像 REINFORCE + group baseline + KL penalty。少了 ratio clipping，短 horizon 任務很容易把 base model 推壞。

4. **trajectory log-prob 有明顯數學不一致**  
   `sample_trajectory()` 在 `t=0` 是 deterministic，不加 noise；但 `get_trajectory_log_prob()` / `backprop_trajectory_loss()` 仍把 `t=0` 當 Normal transition 算 log-prob，而且用了 `alpha[t-1]`，在 `t=0` 會變成 `alpha[-1]`。這會產生錯誤梯度與錯誤 KL。見 [model.py](/home/kingjames23/ReasonMotion/predictor/model/model.py:414)。

5. **log-prob loss 沒有用 missing/future mask，也沒有 normalize 尺度**  
   目前 log-prob 對所有 `K x L` 維度求 sum，包含 observed input frames；但 reward 主要關心 future。這會讓梯度尺度巨大且方向混雜。見 [model.py](/home/kingjames23/ReasonMotion/predictor/model/model.py:478)。

6. **visual jitter 不一定會被目前 smoothness reward 抓住**  
   你現在懲罰的是平均 acceleration magnitude。人眼看到的關節震動常是高頻 jerk / 局部 joint jitter；平均 acceleration 可能沒掉太多，但某些 distal joints 已經在抖。建議加 `jerk`、`velocity-to-GT`、`acceleration-to-GT` 或 high-frequency spectral penalty。

7. **short-term paper 比較可能有 protocol 差異，也可能是 base model 本身不強**  
   你的 `eval_mpjpe.py` 是 single sample、future-only MPJPE，時間映射在 `downsample=2` 下是 25 FPS，80/160/320/400ms 對應 future index 1/3/7/9，這大致合理。差異更可能來自：joint set 17 vs 22、action-wise average vs sample-wise average、是否 deterministic、是否 best-of-N、是否 train output_n=25、以及你的模型本身是 diffusion/imputation style，不一定針對 short-term deterministic MPJPE 最優化。

## 建議優先順序

1. **先修 GRPO log-prob**
   - `get_trajectory_log_prob()` 和 `backprop_trajectory_loss()` 跳過 `t=0`，或讓 sampling 也在 `t=0` 用一致的 stochastic transition。
   - log-prob 只算 future/missing mask。
   - 對 log-prob 依 `num_future_elements * num_steps` normalize。

2. **把 reward 改成 future-aware**
   - `R_GT` 只算 `input_n:`。
   - smoothness 至少分成：boundary smoothness、future acceleration、future jerk。
   - 加 `velocity/acceleration matching to GT`，不要只獎勵「絕對平滑」，否則模型可能變成平滑但不像 GT。

3. **重新做真正 PPO/GRPO**
   - 使用 `ratio = exp(logp_new - logp_old)`。
   - 用 clipped surrogate。
   - `old_logprob` 要來自 rollout 時的 policy 並 detach。
   - KL reference 建議先固定在 SFT/base model，不要每 200 batch 就更新，否則退化會被新的 ref 接受。

4. **重新設計診斷指標**
   - CSV 同時 log `future_mpjpe`、`future_vel_err`、`future_acc_err`、`future_jerk`。
   - smoothness 不只 log reward scalar，也 log raw acceleration/jerk，否則 reward scale 很難判斷。

5. **short-term eval 做 sanity check**
   - 先比較 `pretrained_ckpt` vs RL checkpoints。
   - 同一 protocol 下跑 deterministic 1-sample、multi-seed mean/std。
   - 確認比較 paper 是否同樣是 22 joints、S5 test、25 FPS、80/160/320/400ms、single prediction。