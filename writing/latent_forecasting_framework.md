# Forecasting as Inference on Partially-Observed CRF: Mathematical Framework (v2)

> Encoder-Decoder MFVI with Latent Consistency for Time Series Forecasting

---

## 1. Notation and Problem Setup

**Input / Output:**

- Input time series: $X \in \mathbb{R}^{T \times N}$, where $T$ = time steps, $N$ = number of channels
- Prediction target: $Y \in \mathbb{R}^{S \times N}$, where $S$ = prediction horizon
- After patching with patch length $L_p$:
  - $P = T / L_p$ : number of observed patches
  - $S_f = \lceil S / L_p \rceil$ : number of future patches
  - $\mathbf{x}_{i,t} \in \mathbb{R}^{L_p}$ : the $t$-th patch of channel $i$

**CRF Random Variables:**

- $Z_{i,t}$ : discrete latent label for channel $i$, patch $t$, with label set size $d$
  - $i \in \{1, \ldots, N\}$, $t \in \{1, \ldots, P + S_f\}$
- $H^{(c)}_{i,t}$ : dependency head variable in channel $c$ ($c \in \{1, \ldots, h\}$)
  - Points to another $Z$ that shares the **same channel** (temporal) or **same timestep** (cross-channel)
- $G_{i,t}$ : global topic variable (binary factor, simulates FFN)

---

## 2. Potential Functions

### 2.1 Unary Potential

For **observed** patches ($t \in \{1, \ldots, P\}$):

$$
\phi_u(Z_{i,t}) = \exp\!\left(S^{\text{obs}}_{i,t,Z_{i,t}}\right), \quad S^{\text{obs}}_{i,t} = \text{MLP}_{\text{unary}}(\mathbf{x}_{i,t}) \in \mathbb{R}^d
$$

For **future** patches ($t \in \{P+1, \ldots, P+S_f\}$):

$$
\phi_u(Z_{i,t}) = 1 \quad \text{(i.e., } S^{\text{future}}_{i,t} = \mathbf{0} \in \mathbb{R}^d\text{)}
$$

> **Key design in v2:** Future Z nodes have **zero** unary potential — no local evidence at all. Their posterior is driven **entirely** by messages from the observed Z evidence through ternary factors.

### 2.2 Ternary Potential — Temporal (Same Channel)

For same channel $i$, patches $t$ and $s$:

$$
\phi^{\text{time}}_t\!\left(H^{(c)}_{i,t}, Z_{i,t}, Z_{i,s}\right) =
\begin{cases}
\exp\!\left(T^{(c,\text{time})}_{Z_{i,t}, Z_{i,s}}\right) & \text{if } H^{(c)}_{i,t} = (i, s) \\
1 & \text{otherwise}
\end{cases}
$$

With tensor decomposition:

$$
T^{(c,\text{time})}_{a,b} = \sum_{l=1}^{r} U^{\text{time}}_{a,c,l} \cdot V^{\text{time}}_{b,c,l}
$$

where $r = d/h$ is the decomposition rank, $U^{\text{time}}, V^{\text{time}} \in \mathbb{R}^{d \times h \times r}$.

**v2 distinction: Encoder and decoder maintain separate temporal ternary factors** (see Section 4).

### 2.3 Ternary Potential — Channel (Same Timestep)

For same timestep $t$, channels $i$ and $j$:

$$
\phi^{\text{chan}}_t\!\left(H^{(c)}_{i,t}, Z_{i,t}, Z_{j,t}\right) =
\begin{cases}
\exp\!\left(T^{(c,\text{chan})}_{Z_{i,t}, Z_{j,t}}\right) & \text{if } H^{(c)}_{i,t} = (j, t) \\
1 & \text{otherwise}
\end{cases}
$$

$$
T^{(c,\text{chan})}_{a,b} = \sum_{l=1}^{r} U^{\text{chan}}_{a,c,l} \cdot V^{\text{chan}}_{b,c,l}
$$

Channel ternary factors are **shared** between encoder and decoder.

### 2.4 Binary Potential (FFN-like, G Variables)

$$
\phi_b(Z_{i,t}, G_{i,t}) = \exp\!\left(B_{G_{i,t}, Z_{i,t}}\right), \quad B \in \mathbb{R}^{d_g \times d}
$$

Binary factors are **shared** between encoder and decoder.

---

## 3. MFVI Update Equations

### 3.1 Z Normalization (SquaredSoftmax)

At the start of each MFVI iteration $\tau$:

$$
\tilde{Q}^{(\tau)}_{i,t}(a) = \frac{\left[Q^{(\tau)}_{i,t}(a)\right]^2}{\sum_b \left[Q^{(\tau)}_{i,t}(b)\right]^2}
$$

### 3.2 Message F: Z → H (`calculate_messageF`)

**Temporal direction** — for channel $i$, head channel $c$, position $t$ pointing to $s$ (same channel):

$$
F^{(\tau, \text{time})}_{i,t,c}(s) = \sum_a \sum_b \tilde{Q}^{(\tau)}_{i,t}(a) \cdot \tilde{Q}^{(\tau)}_{i,s}(b) \cdot T^{(c,\text{time})}_{a,b}
$$

Using tensor decomposition + RoPE, in vectorized form:

$$
\mathbf{q}^{u}_{i,t,c} = \text{RoPE}_t\!\left(\tilde{Q}^{(\tau)}_{i,t} \cdot U^{\text{time},(c)}\right) \in \mathbb{R}^r
$$

$$
\mathbf{q}^{v}_{i,s,c} = \text{RoPE}_s\!\left(\tilde{Q}^{(\tau)}_{i,s} \cdot V^{\text{time},(c)}\right) \in \mathbb{R}^r
$$

$$
F^{(\tau, \text{time})}_{i,t,c}(s) = \left\langle \mathbf{q}^{u}_{i,t,c},\; \mathbf{q}^{v}_{i,s,c} \right\rangle
$$

**Channel direction** — identical structure, swapping channel and time indices, using $U^{\text{chan}}, V^{\text{chan}}$.

### 3.3 H Variable Update — Joint Normalization (ST-PT Core Design)

For node $(i,t)$, head channel $c$, the candidate targets include:
- Temporal neighbors: $\{(i,s) : s \neq t\}$
- Channel neighbors: $\{(j,t) : j \neq i\}$

**Concatenate** and **jointly normalize**:

$$
Q^{(\tau)}_{H^{(c)}_{i,t}} = \text{softmax}\!\left(\frac{1}{\lambda_H}\left[F^{\text{time}}_{i,t,c}(\cdot),\; F^{\text{chan}}_{i,t,c}(\cdot)\right]\right)
$$

where $\lambda_H = 1/d$. This creates **competition** between temporal and channel dependencies.

Denote the normalized result as:

$$
Q^{(\tau)}_{H^{(c)}_{i,t}} \to \left(\alpha^{\text{time}}_{i,t,c}(s),\; \alpha^{\text{chan}}_{i,t,c}(j)\right)
$$

### 3.4 Message G: H → Z (`calculate_messageG`)

**Temporal message** (with RoPE inverse rotation `apply_o`):

$$
G^{(\tau,\text{time})}_{i,t}(a) = \sum_c \sum_{s \neq t} \left[\alpha^{\text{time}}_{i,t,c}(s) \cdot \sum_b \tilde{Q}^{(\tau)}_{i,s}(b) \cdot V^{(c,\text{time})}_{b,\cdot} \cdot U^{(c,\text{time})T}_{a,\cdot} \;+\; \alpha^{\text{time}}_{i,s,c}(t) \cdot \sum_b \tilde{Q}^{(\tau)}_{i,s}(b) \cdot U^{(c,\text{time})}_{b,\cdot} \cdot V^{(c,\text{time})T}_{a,\cdot}\right]
$$

**Channel message** — identical structure with channel ternary factors.

### 3.5 Binary Factor Message (FFN-like, `PtTopicModeling`)

$$
G^{(\tau,\text{binary})}_{i,t}(a) = \sum_g \underbrace{\frac{\text{ReLU}\!\left(\sum_b \tilde{Q}^{(\tau)}_{i,t}(b) \cdot B_{g,b}\right)}{\sum_{g'} \text{ReLU}\!\left(\sum_b \tilde{Q}^{(\tau)}_{i,t}(b) \cdot B_{g',b}\right)}}_{Q_G(g)} \cdot B_{g,a}
$$

### 3.6 Z Variable Update

$$
Q^{(\tau+1)}_{i,t}(a) = \frac{1}{\lambda_Z}\left(G^{\text{time}}_{i,t}(a) + G^{\text{chan}}_{i,t}(a) + G^{\text{binary}}_{i,t}(a) + S_{i,t,a}\right)
$$

where $S_{i,t,a}$ is $S^{\text{obs}}_{i,t,a}$ for observed patches or $\mathbf{0}$ for future patches.

### 3.7 Damping (Residual Connection Analog)

$$
Q^{(\tau+1)}_{i,t} \leftarrow \beta \cdot Q^{(\tau+1)}_{i,t} + (1-\beta) \cdot Q^{(\tau)}_{i,t}, \quad \beta = 0.5
$$

---

## 4. Forecasting Pipeline: Encoder-Decoder MFVI v2

### 4.0 Instance Normalization (RevIN)

Before any processing, normalize each input sample:

$$
\mu = \frac{1}{T} \sum_{t} X_t, \quad \sigma = \sqrt{\text{Var}(X) + \epsilon}
$$

$$
\tilde{X} = \frac{X - \mu}{\sigma}
$$

The prediction head output is denormalized: $\hat{Y} = \hat{Y}_{\text{norm}} \cdot \sigma + \mu$.

### 4.1 Phase 1 — Encoder MFVI (Observed Region)

Run $K_{\text{enc}}$ iterations of standard MFVI on the observed graph $\{(i,t) : t \in \{1,\ldots,P\}\}$, using the **encoder temporal ternary factors** $(U^{\text{enc,time}}, V^{\text{enc,time}})$.

**Initialization:**

$$
Q^{(0)}_{i,t} = S^{\text{obs}}_{i,t} = \text{MLP}_{\text{unary}}(\mathbf{x}_{i,t})
$$

**Output:** $\mathbf{Z}^{\text{enc}} = \{Q^{(K_{\text{enc}})}_{i,t}\}_{i,t}$, shape $[B, N, P, d]$.

The H variables only connect within the observed region. $\mathbf{Z}^{\text{enc}}$ **strictly depends only on** $X_{1:T}$.

### 4.2 Phase 2 — Dynamic Prior Generation

Generate an **input-specific** initialization for future Z from the encoder's output:

$$
\bar{\mathbf{Z}}^{\text{enc}} = \frac{1}{P} \sum_{t=1}^{P} \mathbf{Z}^{\text{enc}}_{i,t} \in \mathbb{R}^{[B, N, d]}
$$

$$
\mathbf{Z}^{\text{init}}_{i,t} = \bar{\mathbf{Z}}^{\text{enc}}_i, \quad \forall\, t \in \{P+1, \ldots, P+S_f\}
$$

> **Why mean pooling?** Zero extra parameters. $\mathbf{Z}^{\text{enc}}$ already encodes rich temporal structure via MFVI; the mean is a reasonable global summary. RoPE in the decoder MFVI will differentiate future positions during message passing.

> **Why this matters vs. v1's learnable prior:** In v1, $S^{\text{prior}}$ was context-free (same for all inputs), creating a huge gap to the oracle target. In v2, $\mathbf{Z}^{\text{init}}$ carries observation-specific information. The gap to the oracle is reduced to "has / hasn't seen the future" — exactly what latent consistency should bridge.

### 4.3 Phase 3 — Decoder MFVI (Full Graph, Z_enc Fixed as Evidence)

Extend the graph to $P + S_f$ positions. **Fix** $\mathbf{Z}^{\text{enc}}$ (never updated) — it serves as **pure evidence** that sends messages to future Z.

**Separate decoder temporal ternary factors:**

The decoder uses its own temporal ternary $(U^{\text{dec,time}}, V^{\text{dec,time}})$ while sharing channel ternary $(U^{\text{chan}}, V^{\text{chan}})$ and binary factor $B$ with the encoder.

| Factor | Encoder | Decoder |
|---|---|---|
| Temporal ternary $(U, V)$ | $U^{\text{enc,time}}, V^{\text{enc,time}}$ | $U^{\text{dec,time}}, V^{\text{dec,time}}$ (separate) |
| Channel ternary $(U, V)$ | $U^{\text{chan}}, V^{\text{chan}}$ | $U^{\text{chan}}, V^{\text{chan}}$ (shared) |
| Binary factor $B$ | $B$ | $B$ (shared) |

> **Rationale:** Encoder temporal ternary learns co-occurrence within observations. Decoder temporal ternary learns predictive dynamics from observation to future — fundamentally different relationships, analogous to encoder self-attention vs. decoder cross-attention.

**Unary assignment:**

$$
S^{\text{full}}_{i,t} =
\begin{cases}
S^{\text{obs}}_{i,t} & t \in \{1, \ldots, P\} \\
\mathbf{0} & t \in \{P+1, \ldots, P+S_f\}
\end{cases}
$$

**Initialization:**

$$
Q^{(0, \text{dec})}_{i,t} =
\begin{cases}
\mathbf{Z}^{\text{enc}}_{i,t} & t \in \{1, \ldots, P\} \quad \text{(fixed)} \\
\mathbf{Z}^{\text{init}}_{i,t} & t \in \{P+1, \ldots, P+S_f\} \quad \text{(dynamic prior)}
\end{cases}
$$

**Decoder MFVI loop** ($K_{\text{dec}}$ iterations):

$$
Q^{(\tau+1, \text{dec})} = \text{MFVI}_{\text{dec}}\!\left(S^{\text{full}}, Q^{(\tau, \text{dec})}\right)
$$

$$
Q^{(\tau+1, \text{dec})}_{i,t} \leftarrow \mathbf{Z}^{\text{enc}}_{i,t}, \quad \forall\, t \in \{1, \ldots, P\} \quad \text{(reset observed Z)}
$$

**Output:** $\mathbf{Z}^{\text{dec}} = \{Q^{(K_{\text{dec}})}_{i,t}\}_{t \in \{P+1,\ldots,P+S_f\}}$, shape $[B, N, S_f, d]$.

Key properties:
- $\mathbf{Z}^{\text{enc}}$ **sends messages** to future Z through decoder temporal ternary
- $\mathbf{Z}^{\text{enc}}$ is **never modified** — pure evidence
- Future Z evolves via messages from $\mathbf{Z}^{\text{enc}}$ + messages from other future Z + damping
- Future unary = $\mathbf{0}$, so future Z has **no local bias** — purely message-driven
- RoPE provides positional differentiation across all $P + S_f$ positions

### 4.4 Phase 4 — Prediction Head

Per-patch MLP, shared across all future positions:

$$
\hat{\mathbf{y}}_{i,t} = \text{MLP}_{\text{pred}}(\mathbf{Z}^{\text{dec}}_{i,t}) \in \mathbb{R}^{L_p}
$$

$$
\hat{Y} = \text{reshape}(\hat{\mathbf{y}}) \in \mathbb{R}^{[B, S, N]}
$$

Then apply reverse instance normalization: $\hat{Y} \leftarrow \hat{Y} \cdot \sigma + \mu$.

No cross-patch mixer — MFVI already handles inter-patch dependencies through message passing.

---

## 5. Training Objective

### 5.1 Oracle Target Generation (Training Only)

At training time, we have access to ground truth $Y$. Construct full unary factors for all $P + S_f$ positions and run standard **Encoder** MFVI (using encoder temporal ternary):

$$
S^{\text{oracle}}_{i,t} =
\begin{cases}
S^{\text{obs}}_{i,t} & t \in \{1, \ldots, P\} \\
\text{MLP}_{\text{unary}}(\mathbf{y}_{i,t}) & t \in \{P+1, \ldots, P+S_f\}
\end{cases}
$$

$$
\mathbf{Z}^{\text{target}} = \text{sg}\!\left[\text{MFVI}_{\text{enc}}\!\left(S^{\text{oracle}}, S^{\text{oracle}}, K_{\text{enc}}\right)\right]_{t \in \{P+1,\ldots,P+S_f\}}
$$

Key design choices:
- **Uses encoder ternary** (not decoder ternary): the oracle simulates "what if we observed everything?" — this is the encoder's job
- **$K_{\text{oracle}} = K_{\text{enc}}$**: high-quality target
- **Entirely under `torch.no_grad()`**: oracle is a fixed teacher; stop-gradient prevents collapse
- **`MLP_unary` generalizes naturally**: it sees observed patches during normal training; since future patches are from the same data distribution, generalization is reliable

### 5.2 Latent Consistency Loss (Self-Supervised)

Two complementary terms on future positions only:

**SmoothL1** on raw Z scores (captures absolute scale alignment):

$$
\mathcal{L}_{\text{smooth}} = \frac{1}{N \cdot S_f} \sum_{i,t} \text{SmoothL1}\!\left(\mathbf{Z}^{\text{dec}}_{i,t},\; \text{sg}[\mathbf{Z}^{\text{target}}_{i,t}]\right)
$$

**Cosine similarity** on SquaredSoftmax-normalized distributions (captures distributional shape):

$$
\mathcal{L}_{\text{cos}} = 1 - \frac{1}{N \cdot S_f} \sum_{i,t} \cos\!\left(\psi(\mathbf{Z}^{\text{dec}}_{i,t}),\; \psi(\text{sg}[\mathbf{Z}^{\text{target}}_{i,t}])\right)
$$

where $\psi$ is SquaredSoftmax normalization.

> **Why cosine instead of KL?** SquaredSoftmax outputs are L1-normalized squared values, not proper log-probabilities. KL requires $\log(q)$ which is numerically unstable when $q$ has near-zero entries. Cosine similarity is scale-invariant and widely proven in self-supervised learning (SimCLR, BYOL, VICReg).

### 5.3 Prediction Loss

$$
\hat{Y} = \text{MLP}_{\text{pred}}\!\left(\mathbf{Z}^{\text{dec}}\right), \quad \mathcal{L}_{\text{pred}} = \text{MSE}(\hat{Y},\; Y)
$$

### 5.4 Total Loss

$$
\boxed{\mathcal{L} = \mathcal{L}_{\text{pred}} + \lambda_1 \cdot \mathcal{L}_{\text{smooth}} + \lambda_2 \cdot \mathcal{L}_{\text{cos}}}
$$

Default: $\lambda_1 = 1.0$, $\lambda_2 = 0.5$.

No warmup, no `recon_loss`, no extra tricks.

---

## 6. Complete Training and Inference Procedure

### Training (per batch)

```
Input: (X_{1:T}, Y_{T+1:T+S})

1. Instance norm:
   x = (X - mean(X)) / std(X)
   y_norm = (Y - mean(X)) / std(X)

2. Patching + Unary:
   S_obs = MLP_unary(patch(x))                          [B, N, P, d]

3. Encoder MFVI (K_enc iterations, encoder_iterator):
   Z_enc = MFVI_enc(S_obs, S_obs, K_enc)                [B, N, P, d]

4. Dynamic prior:
   z_init = mean(Z_enc, dim=time).expand(S_f)            [B, N, S_f, d]

5. Decoder MFVI (K_dec iterations, decoder_iterator):
   unary_full = cat([S_obs, zeros])                      [B, N, P+S_f, d]
   z_full = cat([Z_enc, z_init])                         [B, N, P+S_f, d]
   for K_dec iterations:
     z_full = decoder_iterator(unary_full, z_full)
     z_full[:,:,:P,:] = Z_enc                            # reset evidence
   Z_dec = z_full[:,:,P:,:]                              [B, N, S_f, d]

6. Predict:
   Y_hat = reverse_norm(MLP_pred(Z_dec))                 [B, pred_len, N]

7. Oracle (no_grad):
   S_gt = MLP_unary(patch(y_norm))
   S_oracle = cat([S_obs.detach(), S_gt])
   Z_target = MFVI_enc(S_oracle, S_oracle, K_enc)[:,:,P:,:].detach()

8. Loss:
   L = MSE(Y_hat, Y)
     + lambda_1 * SmoothL1(Z_dec, Z_target)
     + lambda_2 * (1 - CosineSim(psi(Z_dec), psi(Z_target)))
```

### Inference

```
Input: X_{1:T}

Steps 1-6 only. No oracle, no latent loss.
```

---

## 7. Parameter Budget

| Component | Parameters | Notes |
|---|---|---|
| `MLP_unary` | $2 \cdot L_p \cdot d + d$ | Shared encoder/oracle |
| Encoder temporal ternary ($U^{\text{enc}}, V^{\text{enc}}$) | $2 d^2$ | Encoder-only |
| **Decoder temporal ternary ($U^{\text{dec}}, V^{\text{dec}}$)** | $2 d^2$ | **New — decoder-only** |
| Channel ternary ($U^{\text{chan}}, V^{\text{chan}}$) | $2 d^2$ | Shared |
| Binary factor $B$ | $d_{\text{ff}} \cdot d$ | Shared |
| `MLP_pred` (per-patch) | $d^2 + d \cdot L_p$ | Per-patch MLP |
| Dynamic prior | **0** | Mean pooling, no params |
| **Total new params vs. ST-PT** | $2 d^2$ | Only the decoder temporal ternary |

For $d = 256$: new params = $2 \times 256^2 =$ **131,072** (reasonable).

---

## 8. Key Design Summary

| Design Choice | Justification | v1 → v2 Change |
|---|---|---|
| Zero future unary | Future Z driven purely by messages from evidence | Replaces learnable $S^{\text{prior}}$ |
| Dynamic prior from $\bar{\mathbf{Z}}^{\text{enc}}$ | Input-specific starting point, small gap to oracle | Replaces context-free learnable prior |
| Separate decoder temporal ternary | Encoder learns co-occurrence, decoder learns prediction | New: separate $(U^{\text{dec}}, V^{\text{dec}})$ |
| Shared channel + binary | Cross-variable structure + FFN are universal | Unchanged |
| Encoder-Decoder separation | Observed Z not affected by future Z | Unchanged |
| Fix Z_enc during decoder | Preserves belief-state causality | Unchanged |
| Cosine similarity loss | More stable than KL for SquaredSoftmax | Replaces KL divergence |
| SmoothL1 + cosine dual loss | Aligns both raw scores and distributional shape | Replaces SmoothL1 + KL |
| Instance normalization (RevIN) | Standard trick for distribution shift | New |
| Stop-gradient on oracle | Prevents representational collapse | Unchanged |

---

## 9. Differentiation from NextLat

| Dimension | NextLat | Our Approach |
|---|---|---|
| **Prediction paradigm** | Autoregressive state transition | Posterior inference on partially-observed PGM |
| **Dynamics source** | External MLP $p_\psi$ | CRF's own ternary factors via message passing |
| **Self-supervised target** | Sees next observation $X_{t+1}$ as input | Infers future Z **without** future observations |
| **Multi-step prediction** | Sequential rollout (accumulative error) | Parallel MFVI (no accumulative error) |
| **Training = Inference?** | No ($p_\psi$ only at training) | Yes (Decoder MFVI used in both) |
| **Extra parameters** | $p_\psi$ MLP | Only decoder temporal ternary |
| **Latent semantics** | Arbitrary continuous vectors | Probability distributions with CRF semantics |
| **Future Z initialization** | Zero / random | Dynamic prior from observed belief states |
