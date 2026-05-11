# TODO


**emergency**
把 short term 用 root-relative 跑一次實驗

- 把 FineFS MoE + RL 訓練好
- 把 H36M short + long 接起來看看 performance
- H36M long RL 可以重 train 一次，因為我有修改 RL 的 code 了 
- 把 H36M 的 GRPO 調整成 single step based，不要一次把整個 trajectory 算出來，這樣太慢了
- 思考看看 GRPO 有什麼更好的 reward design，讓 GRPO 更有其價值
