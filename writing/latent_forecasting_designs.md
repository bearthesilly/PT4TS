# Latent Forecasting 设计方案

## 方案 A：纯 MSE 端到端训练（v2）

### 核心思想

去掉 Oracle 路径和辅助 latent consistency loss，让 Encoder-Decoder MFVI 架构完全由 MSE prediction loss 端到端训练。Decoder ternary factors 直接从预测误差中学习 predictive dynamics。

### 为什么去掉 Oracle

v2 早期版本的 Oracle 设计存在根本性问题：

1. **Target 来自完全不同的推理路径**：Oracle Z_target 来自 encoder MFVI + GT unary（全部位置都有观测），而 Z_dec 来自 decoder MFVI + zero unary（未来位置无观测）。这两个是不同优化问题的解，gap 巨大且无法通过学习 ternary factors 弥合。

2. **移动靶子**：Oracle 使用的 encoder_iterator 和 unary_factors 每步都在变（因为被 prediction loss 训练），导致 Z_target 是一个不断变化的目标，latent loss 难以收敛。

3. **可能与 MSE 冲突**：latent loss 推 Z_dec 逼近 Z_target，但最优的 Z_dec（对于 prediction）不一定等于 Z_target。

### 架构

```
Encoder MFVI（双向，观测区域）→ Z_enc
    ↓
Dynamic Prior: mean(Z_enc) → z_init
    ↓
Decoder MFVI（双向，Z_enc 固定为 evidence）→ Z_dec
    ↓
Per-patch MLP → Y_hat
    ↓
Loss = MSE(Y_hat, Y)
```

### DecoderIterator 设计

Z_enc 作为纯 evidence，不参与更新：

1. 拼接 `[Z_enc, Z_future]` 用于 ternary message 计算
2. 在完整 [P+Sf] 图上计算 head_selection（Z_enc 的 H 变量也参与，因为 `qh[s_obs, t_future]` 贡献给 future Z 的 message）
3. **只提取 future 位置的 messages**
4. Binary (FFN) 只在 future Z 上计算
5. 更新公式：`z_future_new = (m_t + m_c + m_g) / regularize_z`（无 unary 项）

### 梯度流

$$\text{MSE} \xrightarrow{\partial} \text{predictor} \xrightarrow{\partial} Z_{\text{dec}} \xrightarrow{\partial} \text{decoder ternary} + Z_{\text{enc}} \xrightarrow{\partial} \text{encoder ternary} + \text{unary MLP}$$

所有组件都从 MSE 获得梯度。v15 的 encoder ternary 也是纯 MSE 训练的，所以 decoder ternary 没有理由学不出来。

### 关键优点

- **训练 = 推理**：零 distribution shift
- **简洁**：无 Oracle、无辅助 loss、无 warmup
- **直接优化**：decoder ternary 被直接优化为"产出让预测最准的 Z_dec"

---

## 方案 C：Causal MFVI + NextLat（v3）

### 核心思想

将 ST-PT 的 encoder MFVI 改为 causal 版本，使得 $Z_{\text{enc}}[t]$ 成为真正的 belief state（只依赖 $X_{1:t}$）。然后用 NextLat 式的自监督 loss 训练 encoder，使其隐状态捕捉 latent dynamics。Decoder MFVI 保持双向，从 belief states 推理未来。

### 为什么需要 Causal MFVI

NextLat 的前提是隐状态 $h_t$ 是 belief state：

$$\mathbb{E}[f(X_{t+1:T}) \mid h_t] = \mathbb{E}[f(X_{t+1:T}) \mid X_{1:t}]$$

但 ST-PT 的 MFVI 是双向的：$Z[t]$ 通过 ternary message passing 看到了 $X_1$ 到 $X_P$ 的全部信息。$Z[3]$ 已经"知道" $Z[10]$ 的内容，"从 $Z[t]$ 预测 $Z[t+1]$" 变成 trivial 的任务。

**必须让 encoder MFVI 变成 causal 的，NextLat 才有意义。**

### Causal MFVI 的数学

#### Temporal Causal Mask

$$\text{mask}_{\text{causal}}[t, s] = \begin{cases} 1 & s < t \\ 0 & s \geq t \end{cases}$$

#### messageG 的 Part 2 问题

标准 messageG 有两部分：

- **Part 1（前向流）**：$G^{(1)}_t = \sum_{s} \alpha_{t,s} \cdot (\tilde{Q}_s V^T) U$ — Z[t] 的 H 指向 Z[s]
- **Part 2（反向流）**：$G^{(2)}_t = \sum_{s} \alpha_{s,t} \cdot (\tilde{Q}_s U^T) V$ — Z[s] 的 H 指向 Z[t]

即使 causal mask 让 $\alpha_{t,s} = 0$ for $s \geq t$，Part 2 中的 $\alpha_{s,t}$ 对 $s > t$ **不为零**（因为 $s$ 可以 causal 地关注 $t < s$）。这导致 Z[t] 收到来自未来 Z[s] 的信息——**因果性被破坏**。

**解决方案**：Temporal 方向只保留 Part 1，去掉 Part 2：

$$G^{\text{causal,time}}_t = \sum_{s < t} \alpha_{t,s} \cdot (\tilde{Q}_s V^T) U$$

这等价于标准 causal Transformer 的 attention output。Channel 方向保持完全双向（Part 1 + Part 2）。

#### 多轮迭代下的因果性保证

- 第 1 轮：$Z^{(1)}[i, t]$ 依赖 $\{Z^{(0)}[i, s] : s < t\}$（temporal）+ $\{Z^{(0)}[j, t] : j \neq i\}$（channel）
- 第 K 轮：$Z^{(K)}[i, t]$ 依赖所有 channel 在 $s \leq t$ 时刻的信息
- **因果不变量始终成立**

### NextLat Loss

#### Predictor $f_\psi$

Per-channel 共享 MLP：$\hat{Z}_{i, t+1} = f_\psi(Z_{i, t})$

#### Loss

$$\mathcal{L}_{\text{NextLat}} = \frac{1}{N(P-1)} \sum_{i,t} \text{SmoothL1}(\text{sg}[Z_{i,t+1}], f_\psi(Z_{i,t}))$$

**为什么这比 v2 的 Oracle loss 好**：

| 维度 | v2 Oracle | v3 NextLat |
|---|---|---|
| Target 来源 | 不同推理路径（encoder + GT unary） | **同一推理路径**（同一 causal encoder） |
| Target 和 prediction 的 gap | 巨大（有/无 GT unary 的区别） | **小且有意义**（相邻时间步的自然差异） |
| 是否 trivial | N/A | **非 trivial**（causal MFVI 让 Z[t] 不知道 Z[t+1] 的内容） |
| 移动靶子 | 是（encoder 每步变） | **也会变，但 gap 小得多** |

### 完整 Pipeline

```
训练:
  1. Causal Encoder MFVI（causal temporal mask + 只有 Part 1）→ Z_enc
     Z_enc[t] 是 X_{1:t} 的 belief state
  2. NextLat loss: predict Z_enc[t+1] from Z_enc[t]
     L_next = SmoothL1(sg[Z_enc[t+1]], f_ψ(Z_enc[t]))
  3. Dynamic prior: mean(Z_enc) → z_init
  4. Decoder MFVI（双向，Z_enc 固定为 evidence）→ Z_dec
  5. Per-patch MLP → Y_hat
  6. Total loss = MSE(Y_hat, Y) + λ · L_next

推理:
  步骤 1, 3, 4, 5。不计算 NextLat loss。
```

### 两个 Loss 的互补关系

- **MSE** → 训练 decoder ternary + predictor，优化预测精度
- **NextLat** → 训练 encoder，使 Z_enc 成为好的 belief state（表示质量）

NextLat 不直接帮助预测，但它提升了 Z_enc 的质量，间接帮助 decoder 从更好的 evidence 出发进行推理。

### 参数开销

| 组件 | 参数量 | 说明 |
|---|---|---|
| Causal encoder (temporal + channel ternary, binary) | 和 v15 相同 | encoder 本体 |
| Decoder (temporal ternary) | $2d^2$ | 独有的 decoder temporal ternary |
| NextLat predictor $f_\psi$ | $2d^2 + 2d$ | 两层 MLP |
| 总新增（相比 v15） | $4d^2 + 2d$ | d=128 时约 66K |

### 理论合理性

1. Causal MFVI 产出 belief state：$Z[t]$ 只依赖 $X_{1:t}$
2. NextLat loss 有意义：$Z[t]$ 不包含 $X_{t+1}$ 的信息，预测 $Z[t+1]$ 是非 trivial 的
3. $f_\psi$ 只在训练时使用：和 NextLat 论文一致
4. Decoder 保持双向合理：Decoder 是从 evidence 推理未来，不需要 causal
5. 两个 loss 互补：NextLat 提升表示，MSE 优化预测

---

## 方案对比总结

| 维度 | v2 (Plan A) | v3 (Plan C) |
|---|---|---|
| Encoder | 双向 MFVI | **Causal MFVI** |
| Z_enc 语义 | 后验（给定全部观测） | **Belief state（给定 $X_{1:t}$）** |
| 自监督信号 | 无 | **NextLat loss** |
| Decoder | 双向 DecoderIterator | 双向 DecoderIterator |
| Prediction loss | MSE only | MSE + λ·NextLat |
| 训练 ≈ 推理 | 完全一致 | 几乎一致（推理不算 NextLat loss） |
| 新增参数 | $2d^2$ (decoder ternary) | $4d^2 + 2d$ (decoder ternary + $f_\psi$) |
| 复杂度 | 最简 | 中等 |
| 理论动机 | 端到端 MSE 足够 | CRF belief state + NextLat dynamics |

### 实现文件

| 方案 | 模型文件 | 实验脚本 |
|---|---|---|
| Plan A (v2) | `models/PT_forecast_latent_v2.py` | `scripts/.../PT_latent_v2_all_seed2021.sh` |
| Plan C (v3) | `models/PT_forecast_latent_v3.py` | `scripts/.../PT_latent_v3_all_seed2021.sh` |

---

## 方案 D：Channel-Independent Encoder-Decoder MFVI（v4）

### 核心思想

借鉴 PatchTST 和 TimeMixer 的 channel independence 策略：每个 channel 独立运行一套 PT（共享权重），不做跨 channel 交互。

很多时序 benchmark 的 channel 间相关性弱，跨 channel 建模反而引入噪声（"noisy channel pollution"）。Channel independence 通过去除这些不可靠的交互来提升性能。

### 架构

```
输入 [B, T, N] → reshape 为 [B*N, T, 1]（每个 channel 作为独立样本）
    ↓
Per-channel Instance Norm
    ↓
Patching + Unary MLP（共享权重）
    ↓
Encoder MFVI（共享权重，N_eff=1，channel ternary 自动为零）
    ↓
Dynamic Prior: mean(Z_enc)
    ↓
Decoder MFVI（共享权重，Z_enc 固定为 evidence）
    ↓
Per-patch MLP predictor（共享权重）
    ↓
reshape 回 [B, pred_len, N] + 反归一化
```

### 实现细节

- 构造 iterator 时设 `enc_in=1`
- `PtHeadSelection` 中 channel 方向只有 1 个位置（自身），对角线被 mask → channel attention 权重全为 0
- Joint H normalization 中 channel 候选为 $-\infty$ → softmax 自动把全部权重给 temporal 候选
- **效果等价于纯 temporal MFVI**，无需修改任何 iterator 代码

### 和 PatchTST / TimeXer 的对比

| 模型 | Encoder 权重 | 预测头权重 | Per-channel 参数 |
|---|---|---|---|
| PatchTST | 完全共享 | 完全共享 | 零 |
| TimeXer | 完全共享 | 完全共享 | `glb_token [N, d]` |
| **v4** | 完全共享 | 完全共享 | **零** |

### 关键优点

1. **避免 noisy channel pollution**：弱相关的 channel 不会互相干扰
2. **参数不变**：和 v2 完全相同的参数量（ternary/binary 大小不依赖 `enc_in`）
3. **隐式加速**：channel ternary 计算为零，只有 temporal MFVI 有效
4. **训练 = 推理**：纯 MSE，无辅助 loss

### 可能的扩展：Channel Embedding

如果需要让模型感知"这是哪个 channel"，可加轻量的 channel embedding：

$$\text{unary}_{i,t} = \text{MLP}_{\text{unary}}(x_{i,t}) + e_i, \quad e_i \in \mathbb{R}^d$$

仅增加 $N \times d$ 个参数（N=7, d=128 时为 896 个参数）。

---

## 方案 E：Causal MFVI + AR Rollout + Decoder MFVI 纠正（v5）

### 核心思想

将 NextLat 的隐空间自回归和 PT 的 CRF 推理结合：

1. **Causal encoder** 产出 belief states
2. **Residual AR rollout** 从最后一个 belief state 逐步外推，提供 dynamics-informed 初始化
3. **Decoder MFVI** 在 AR 初始化的基础上做并行全局纠正，修复累积误差

### 为什么隐空间 AR 可行？

传统 AR 在**观测空间**自回归——预测 $\hat{X}_{t+1}$，喂回 encoder 预测 $\hat{X}_{t+2}$——误差在 encode-decode 循环中快速放大。

我们的 AR 在**隐空间**自回归——$Z_{P+1} = f_\psi(Z_P)$, $Z_{P+2} = f_\psi(Z_{P+1})$...——**永远不把预测结果喂回 encoder**。误差只在隐空间累积，不经过 observation-space 的 re-encoding。

PT 框架的额外优势：
- Z 是概率分布（SquaredSoftmax 归一化），比 raw vector 更平滑稳定
- Causal MFVI 产出的 Z 是 belief state，dynamics 应该比观测空间更规则
- Residual transition $f_\psi(z) = z + \text{MLP}(z)$ 保证小增量变化

### Residual Transition 设计

$$f_\psi(Z_t) = Z_t + \text{MLP}(Z_t)$$

残差连接确保：
- 即使 MLP 输出不好，结果也接近输入
- 相邻 Z 之间的变化是小增量
- 长 rollout 不会快速偏移

### NextLat 训练（1-step + 2-step）

在观测序列内训练 transition：

$$\mathcal{L}_{\text{1-step}} = \text{SmoothL1}(\text{sg}[Z_{t+1}], f_\psi(Z_t))$$

$$\mathcal{L}_{\text{2-step}} = \text{SmoothL1}(\text{sg}[Z_{t+2}], f_\psi(f_\psi(Z_t)))$$

$$\mathcal{L}_{\text{NextLat}} = \mathcal{L}_{\text{1-step}} + 0.5 \cdot \mathcal{L}_{\text{2-step}}$$

Multi-step 训练迫使 $f_\psi$ 学会稳定的长期 rollout，而非只优化 1 步准确。

### AR Rollout + MFVI 纠正 = 最优组合

**当前 v2/v3/v4 的 decoder 初始化**：`mean(Z_enc)` → 对所有 future patch 都相同，无 dynamics 信息。Decoder ternary 必须从零开始推理。

**v5 的 decoder 初始化**：AR rollout → 每个 future patch 有不同的、dynamics-informed 的起点。Decoder MFVI 只需微调。

类比：
- v2/v4：给你一张白纸，画出完整的未来 → 难
- **v5：给你一个草稿（AR），精修即可 → 容易得多**

MFVI 纠正的作用：
- AR rollout 的累积误差由 MFVI 的**并行全局推理**修复
- MFVI 能看到 Z_enc（真实 evidence）和所有 future Z 的全局结构
- 修复 AR 的局部累积偏差

### 完整 Pipeline

```
训练:
  1. Causal Encoder MFVI → Z_enc [B, N, P, d]（belief states）
  2. NextLat loss（1-step + 2-step，观测序列内）
  3. AR Rollout: Z_{P+k} = Z_{P+k-1} + MLP(Z_{P+k-1}), k=1..Sf
  4. Decoder MFVI（Z_enc evidence + AR init）→ Z_dec
  5. Per-patch MLP → Y_hat
  6. Loss = MSE(Y_hat, Y) + λ · L_NextLat

推理:
  完全一样（步骤 1, 3, 4, 5）
```

### 梯度流

$$\text{MSE} \xrightarrow{\partial} \text{predictor} \xrightarrow{\partial} Z_{\text{dec}} \xrightarrow[\text{MFVI 纠正}]{\partial} Z_{\text{future\_init}} \xrightarrow[\text{AR rollout}]{\partial} f_\psi + Z_{\text{enc}} \xrightarrow{\partial} \text{encoder}$$

$$\mathcal{L}_{\text{NextLat}} \xrightarrow{\partial} f_\psi + Z_{\text{enc}} \xrightarrow{\partial} \text{encoder}$$

MSE 的梯度经过 decoder MFVI → AR rollout → transition → encoder，全链路可微。NextLat loss 额外强化 encoder 的 belief state 质量和 transition 的 dynamics 准确度。

### 参数开销

| 组件 | 参数量 | 说明 |
|---|---|---|
| Causal encoder ternary | 和 v15 相同 | |
| Decoder ternary | $2d^2$ | 独有的 decoder temporal |
| **Residual Transition $f_\psi$** | $2d^2 + 2d$ | **新增** |
| 总新增（相比 v15） | $4d^2 + 2d$ | d=128 时约 66K |

---

## 全版本对比总结

| 版本 | Encoder | 初始化 | Decoder | 自监督 | Channel | 特色 |
|---|---|---|---|---|---|---|
| v2 | 双向 MFVI | mean(Z_enc) | 双向 DecoderIter | 无 | 跨 channel | 纯 MSE 端到端 |
| v3 | **Causal** MFVI | mean(Z_enc) | 双向 DecoderIter | **NextLat** | 跨 channel | Belief state + NextLat |
| v4 | 双向 MFVI | mean(Z_enc) | 双向 DecoderIter | 无 | **独立** | Channel independence |
| **v5** | **Causal** MFVI | **AR rollout** | 双向 DecoderIter | **NextLat multi-step** | 跨 channel | AR + MFVI 纠正 |

### 实现文件

| 方案 | 模型文件 | 实验脚本 |
|---|---|---|
| Plan A (v2) | `models/PT_forecast_latent_v2.py` | `scripts/.../PT_latent_v2_all_seed2021.sh` |
| Plan C (v3) | `models/PT_forecast_latent_v3.py` | `scripts/.../PT_latent_v3_all_seed2021.sh` |
| Plan D (v4) | `models/PT_forecast_latent_v4.py` | `scripts/.../PT_latent_v4_all_seed2021.sh` |
| Plan E (v5) | `models/PT_forecast_latent_v5.py` | `scripts/.../PT_latent_v5_all_seed2021.sh` |
