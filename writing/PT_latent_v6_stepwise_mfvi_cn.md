# PT_latent_v6：单步 AR Decoder 的 MFVI 公式、代码对应与整体流程

## 1. 这一版模型到底在做什么

`PT_forecast_latent_v6` 不再像 `v5` 那样先把整段 future latent 一次性 rollout 出来，再并行修正。

它改成了真正的 **step-wise autoregressive decoding**：

1. 先对历史 patch 做一次 causal MFVI，得到历史 belief states。
2. 用当前最后一个 latent 预测下一 patch。
3. 把这个 patch 编码成 unary evidence。
4. 在“历史 Q 固定为 evidence”的前提下，只对新时间片 `t+1` 做一次单步 MFVI，得到新的 latent。
5. 再用这个新的 latent 继续预测下一 patch。

训练时会同时跑一条 teacher-forced future latent 路径，作为 latent GT。

---

## 2. DecStep 的图模型定义

设当前已经有历史/前缀的固定状态：

\[
Q_{1:t}^{\text{past}}=\{Q_{i,s}\mid i=1,\dots,N,\ s=1,\dots,t\}
\]

这里：

- \(i\) 表示通道
- \(s\) 表示 patch 时间步
- \(Q_{i,s}\in \Delta^d\) 是已经推断好的 latent posterior

现在我们要推断的是新时间片：

\[
Z_{i,t+1}, \quad i=1,\dots,N
\]

对这组新变量，图里有三类信息源：

1. **历史 evidence**
   - 同一通道上的过去状态 \(Q_{i,1:t}\)
   - 这些状态是固定 evidence，不再更新

2. **当前观测 evidence**
   - 当前 patch \(y_{i,t+1}\)
   - 通过 unary encoder 变成 \(U_{i,t+1}\in\mathbb{R}^d\)

3. **当前切片内部的 channel 关系**
   - 新时间片内部各通道 \(Z_{1:N,t+1}\) 之间相互作用

因此，`DecStep` 不是在更新整段未来，而是在更新一张很小的子图：

- past `Q` 只作为固定证据
- current `Z_{:,t+1}` 是待推断变量
- `H` 变量表示当前节点的 head 选谁

---

## 3. 单步 MFVI 的数学公式

### 3.1 Unary evidence：\(y_{t+1}\) 如何进入

当前 patch 先经 MLP 得到 unary potential：

\[
U_{i,t+1} = f_{\text{unary}}(y_{i,t+1}) \in \mathbb{R}^d
\]

对应势函数：

\[
\phi_u(Z_{i,t+1}=a)=\exp(U_{i,t+1}(a))
\]

这就是“`y_{t+1}` 作为 evidence”的精确定义。

---

### 3.2 时间方向的 head logits：历史 \(Q\) 如何作为 evidence 传递信息

对当前节点 \((i,t+1)\)，时间方向候选 head 是同一通道上的过去节点：

\[
(i,1),\dots,(i,t)
\]

第 \(c\) 个 dependency head channel 下，时间方向的 head logit 为：

\[
F^{\text{time},(\ell)}_{i,c}(s)
=
\sum_a \sum_b
q^{(\ell)}_{i,t+1}(a)\,Q_{i,s}(b)\,T^{\text{time},(c)}_{a,b},
\quad s=1,\dots,t
\]

注意：

- \(q^{(\ell)}_{i,t+1}\) 是当前待推断变量在第 \(\ell\) 轮的分布
- \(Q_{i,s}\) 是过去的 fixed evidence
- past 不更新，所以这里只存在 `current -> past` 的兼容性计算，不存在 future 反向改 past

---

### 3.3 当前切片 channel 方向的 head logits

同一步内部，各通道之间也竞争：

\[
F^{\text{chan},(\ell)}_{i,c}(j)
=
\sum_a \sum_b
q^{(\ell)}_{i,t+1}(a)\,q^{(\ell)}_{j,t+1}(b)\,T^{\text{chan},(c)}_{a,b},
\quad j=1,\dots,N,\ j\neq i
\]

---

### 3.4 时间 + 通道联合归一化

和 ST-PT / 现有 PT 实现一致，这里不是两个独立 softmax，而是把时间候选和通道候选拼起来后做 **一次联合 softmax**：

\[
\alpha^{(\ell)}_{i,c}
=
\operatorname{softmax}
\left(
\frac{
\left[
F^{\text{time},(\ell)}_{i,c}(1:t),
F^{\text{chan},(\ell)}_{i,c}(1:N)
\right]
}{\lambda_H}
\right)
\]

其中对 \(j=i\) 的 channel 自环做 mask。

拆开记为：

\[
\alpha^{\text{time},(\ell)}_{i,c}(s), \qquad
\alpha^{\text{chan},(\ell)}_{i,c}(j)
\]

---

### 3.5 `H -> Z` 的消息回传

#### 时间消息

由于过去节点是 fixed evidence，所以时间消息只有“从过去到当前”的一项：

\[
G^{\text{time},(\ell)}_i(a)
=
\sum_c \sum_{s=1}^{t}
\alpha^{\text{time},(\ell)}_{i,c}(s)
\sum_b
Q_{i,s}(b)\,T^{\text{time},(c)}_{a,b}
\]

这里没有
\(\alpha_{s,c}(t+1)\)
那一项，因为 past 不更新，也不存在 current 成为 past 的 head 再反过来更新 past 的情形。

#### 通道消息

当前切片内部的 channel 是 jointly inferred，所以这里保留双向项：

\[
G^{\text{chan},(\ell)}_i(a)
=
\sum_c \sum_{j\neq i}
\Big[
\alpha^{\text{chan},(\ell)}_{i,c}(j)
\sum_b q^{(\ell)}_{j,t+1}(b) T^{\text{chan},(c)}_{a,b}
\;+\;
\alpha^{\text{chan},(\ell)}_{j,c}(i)
\sum_b q^{(\ell)}_{j,t+1}(b) T^{\text{chan},(c)}_{b,a}
\Big]
\]

#### Global / FFN-like 消息

\[
G^{g,(\ell)}_i = \text{TopicModeling}(q^{(\ell)}_{i,t+1})
\]

---

### 3.6 Z 的更新

于是当前 slice 的 posterior logits 更新为：

\[
\eta^{(\ell+1)}_{i,t+1}
=
U_{i,t+1}
G^{\text{time},(\ell)}_i
G^{\text{chan},(\ell)}_i
G^{g,(\ell)}_i
\]

再做 PT 里对应的归一化：

\[
q^{(\ell+1)}_{i,t+1}
=
\operatorname{Norm}(\eta^{(\ell+1)}_{i,t+1})
\]

代码里还保留了 damping：

\[
\eta^{(\ell+1)}_{i,t+1}
\leftarrow
\frac{1}{2}\eta^{(\ell+1)}_{i,t+1}
\;+\;
\frac{1}{2}\eta^{(\ell)}_{i,t+1}
\]

初始值就是 unary：

\[
\eta^{(0)}_{i,t+1}=U_{i,t+1}
\]

---

## 4. 代码是否符合上面的 MFVI 流程

结论先说：**当前 `v6` 的 `DecStep` 实现，和上面的单步 MFVI 计算流程是对齐的。**

下面逐项对应。

### 4.1 当前观测 \(y_{t+1}\) 进入 unary

代码：

- `patch_pred = self.patch_predictor(z_prev_student)`
- `unary_student = self.unary_factors(patch_pred)`
- teacher 路径则是 `unary_teacher = self.unary_factors(future_gt[:, :, step, :])`

对应数学式：

\[
U_{i,t+1} = f_{\text{unary}}(y_{i,t+1})
\]

也就是说：

- student 用预测 patch 当 evidence
- teacher 用 GT patch 当 evidence

---

### 4.2 历史 \(Q\) 作为 fixed evidence

代码里没有把历史 `Q` 再放回完整图做更新，而是先把历史状态投影成时间方向的 cached value：

- `hist_time_v = self.decoder._project_time_values_sequence(hist_norm, hist_time_pe)`
- 后续 decoder step 只读 `past_time_v`

这和数学上“过去的 \(Q_{i,1:t}\) 是 fixed evidence”是一致的。

并且这是 memory-friendly 的关键：

- 不需要存整段 future 图
- 也不需要每一步都重算 past 的 time projection

---

### 4.3 时间 head logits

代码：

- `time_q_u = self._project_time_query_step(q_new_norm, time_pe_now)`
- `logits_time = torch.einsum("bnhr,bnhlr->bnhl", time_q_u, past_time_v).permute(0, 2, 1, 3)`

这正是：

\[
F^{\text{time}}_{i,c}(s)
=
\sum_a \sum_b q_{i,t+1}(a)\,Q_{i,s}(b)\,T^{\text{time},(c)}_{a,b}
\]

的张量化实现。

---

### 4.4 channel head logits

代码：

- `mF_c, ... = self.head_selection._messageF(q_new_norm, channel_mask, channel_pe, ...)`

这正是当前切片内部的：

\[
F^{\text{chan}}_{i,c}(j)
=
\sum_a \sum_b q_{i,t+1}(a)\,q_{j,t+1}(b)\,T^{\text{chan},(c)}_{a,b}
\]

---

### 4.5 联合 softmax

代码：

- `combined = softmax(cat([logits_time, mF_c]) / self.regularize_h)`
- `qh_time, qh_channel = split(...)`

这对应：

\[
\alpha_{i,c}
=
\operatorname{softmax}
\left(
\frac{[F^{\text{time}}_{i,c},F^{\text{chan}}_{i,c}]}{\lambda_H}
\right)
\]

所以时间依赖和通道依赖确实是在竞争同一个 probability mass。

---

### 4.6 时间消息回传

代码：

- `time_msg = einsum("bhnl,bnhlr->bnhr", qh_time, past_time_v)`
- 再经 output RoPE 和 `ternary_factor_u_time` 回到 `d` 维

这对应：

\[
G^{\text{time}}_i(a)
=
\sum_c \sum_s \alpha^{\text{time}}_{i,c}(s)\sum_b Q_{i,s}(b)T^{\text{time},(c)}_{a,b}
\]

这里确实只有单向 past -> current，没有反向项。  
这正是“历史 `Q` 是 evidence”的应有形式。

---

### 4.7 channel 消息回传

代码：

- `channel_msg = self.head_selection._messageG_full(...)`

这保留了当前 slice 内的双向 channel 作用，符合：

\[
G^{\text{chan}}_i(a)
=
\sum_c \sum_{j\neq i}
\Big[
\alpha_{i,c}(j)\sum_b q_j(b)T_{a,b}
\;+\;
\alpha_{j,c}(i)\sum_b q_j(b)T_{b,a}
\Big]
\]

---

### 4.8 unary + messages 更新

代码：

- `q_new = (time_msg + channel_msg + global_msg + unary_current) / regularize_z`
- `q_new = 0.5 * (q_new + old_q)`

这正是：

\[
\eta^{(\ell+1)} = U + G^{time} + G^{chan} + G^g
\]

加上 damping。

---

## 5. 需要明确说明的一点

当前 `v6` 的 `DecStep` 是一个 **单步后验更新器**：

- 历史 `Q` 作为 fixed evidence
- 当前 patch `y_{t+1}` 作为 unary evidence
- 当前 slice 内通道 jointly infer

所以它不是：

- “一个 `q_P` 直接 rollout 全 horizon latent”

而是：

- “prefix latent -> next patch -> unary evidence -> next latent posterior”

这和 transformer decoding 的思路是一致的。

---

## 6. 整个 v6 的训练与推理流程

### 6.1 历史编码

\[
Q_{1:P}^{hist} = \mathrm{CausalMFVI}(X_{1:P})
\]

### 6.2 Student AR 路径

对每个 future step：

1. 用当前最后一个 latent 预测下一 patch
\[
\hat y_{t+1}=g(z_t)
\]

2. 把它变成 unary
\[
U_{t+1}^{st}=f_{\text{unary}}(\hat y_{t+1})
\]

3. 做单步 `DecStep`
\[
Q_{t+1}^{st}=\mathrm{DecStep}(Q_{1:t}^{st}\ \text{fixed},\ U_{t+1}^{st})
\]

### 6.3 Teacher-forced latent GT 路径

训练时同步做：

\[
U_{t+1}^{tf}=f_{\text{unary}}(y_{t+1}^{gt})
\]

\[
Q_{t+1}^{tf}=\mathrm{DecStep}(Q_{1:t}^{tf}\ \text{fixed},\ U_{t+1}^{tf})
\]

### 6.4 辅助损失

当前实现用的是 latent KL：

\[
\mathcal L_{latent}
=
\sum_{t=P+1}^{P+S_f}
\mathrm{KL}\big(
\operatorname{sg}(Q_t^{tf})
\;\|\;
Q_t^{st}
\big)
\]

总代价仍然是：

\[
\mathcal L
=
\mathcal L_{pred}
\lambda \mathcal L_{latent}
\]

---

## 7. 当前实现的优点

1. **数学语义更干净**
   - history 是 evidence
   - current patch 是 unary evidence
   - current latent 是 posterior

2. **更接近 transformer decoding**
   - 先预测 next patch
   - 再把 next patch 喂回来
   - 再更新 next latent

3. **更省显存**
   - 不构造全 future 图
   - teacher 路径 `no_grad`
   - past 只缓存 time-value projection

---

## 8. 一句话总结

`v6` 的 `DecStep` 本质上是在做：

> **给定固定的 prefix latent evidence，以及当前 patch 的 unary evidence，对新时间片做一次局部 MFVI posterior inference。**

这和我这版代码的实现是一致的。
