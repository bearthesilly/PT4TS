# Forecasting as Inference on Partially-Observed CRF: Mathematical Framework

> Encoder-Decoder MFVI with Latent Consistency for Time Series Forecasting

---

## 1. Notation and Problem Setup

**Input / Output:**

- Input time series: $X \in \mathbb{R}^{T \times N}$, where $T$ = time steps, $N$ = number of channels
- Prediction target: $Y \in \mathbb{R}^{S \times N}$, where $S$ = prediction horizon
- After patching with patch length $L_p$:
  - $P = T / L_p$ : number of observed patches
  - $S_p = S / L_p$ : number of future patches
  - $\mathbf{x}_{i,t} \in \mathbb{R}^{L_p}$ : the $t$-th patch of channel $i$

**CRF Random Variables:**

- $Z_{i,t}$ : discrete latent label for channel $i$, patch $t$, with label set size $d$
  - $i \in \{1, \ldots, N\}$, $t \in \{1, \ldots, P + S_p\}$
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

For **future** patches ($t \in \{P+1, \ldots, P+S_p\}$):

$$
\phi_u(Z_{i,t}) = \exp\!\left(S^{\text{prior}}_{Z_{i,t}}\right), \quad S^{\text{prior}} \in \mathbb{R}^d \quad \text{(learnable prior parameter)}
$$

> **Key distinction from standard ST-PT:** Future Z nodes have no observation-driven unary factor; only a learnable prior.

### 2.2 Ternary Potential — Temporal (Same Channel)

For same channel $i$, patches $t$ and $s$:

$$
\phi^{\text{time}}_t\!\left(H^{(c)}_{i,t}, Z_{i,t}, Z_{i,s}\right) =
\begin{cases}
\exp\!\left(T^{(c,\text{time})}_{Z_{i,t}, Z_{i,s}}\right) & \text{if } H^{(c)}_{i,t} = (i, s) \\
1 & \text{otherwise}
\end{cases}
$$

With tensor decomposition (corresponding to `ternary_factor_u_time`, `ternary_factor_v_time`):

$$
T^{(c,\text{time})}_{a,b} = \sum_{l=1}^{r} U^{\text{time}}_{a,c,l} \cdot V^{\text{time}}_{b,c,l}
$$

where $r = d/h$ is the decomposition rank, $U^{\text{time}}, V^{\text{time}} \in \mathbb{R}^{d \times h \times r}$.

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

### 2.4 Binary Potential (FFN-like, G Variables)

$$
\phi_b(Z_{i,t}, G_{i,t}) = \exp\!\left(B_{G_{i,t}, Z_{i,t}}\right), \quad B \in \mathbb{R}^{d_g \times d}
$$

---

## 3. MFVI Update Equations

The following equations apply to both Encoder and Decoder phases. The **only difference** is whether the unary factor is observation-driven ($S^{\text{obs}}$) or a learnable prior ($S^{\text{prior}}$).

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

where $S_{i,t,a}$ is $S^{\text{obs}}_{i,t,a}$ for observed patches or $S^{\text{prior}}_a$ for future patches.

### 3.7 Damping (Residual Connection Analog)

$$
Q^{(\tau+1)}_{i,t} \leftarrow \beta \cdot Q^{(\tau+1)}_{i,t} + (1-\beta) \cdot Q^{(\tau)}_{i,t}, \quad \beta = 0.5
$$

---

## 4. Forecasting Pipeline: Encoder-Decoder MFVI

### 4.1 Phase 1 — Encoder MFVI (Observed Region)

Run $K_{\text{enc}}$ iterations of standard MFVI on the observed graph $\{(i,t) : t \in \{1,\ldots,P\}\}$.

**Initialization:**

$$
Q^{(0)}_{i,t} = S^{\text{obs}}_{i,t} = \text{MLP}_{\text{unary}}(\mathbf{x}_{i,t})
$$

**Output:** $\mathbf{Z}^{\text{enc}} = \{Q^{(K_{\text{enc}})}_{i,t}\}_{i,t}$, shape $[B, N, P, d]$.

The H variables only connect within the observed region. $\mathbf{Z}^{\text{enc}}$ **strictly depends only on** $X_{1:T}$.

### 4.2 Phase 2 — Decoder MFVI (Future Region)

Extend the graph to $P + S_p$ positions. **Fix** $\mathbf{Z}^{\text{enc}}$ (not updated during decoder iterations).

**Initialization:**

$$
Q^{(0, \text{dec})}_{i,t} =
\begin{cases}
\mathbf{Z}^{\text{enc}}_{i,t} & t \in \{1, \ldots, P\} \quad \text{(fixed, not updated)} \\
S^{\text{prior}} & t \in \{P+1, \ldots, P+S_p\} \quad \text{(learnable prior)}
\end{cases}
$$

Run $K_{\text{dec}}$ iterations. **Key modifications:**

**(a)** Message F covers $P + S_p$ positions. Observed Z uses fixed $\mathbf{Z}^{\text{enc}}$.

**(b)** For future $Z_{i,t}$ ($t > P$), its H variable points to:
- Temporal: $\{(i,s) : s \in \{1,\ldots,P+S_p\},\; s \neq t\}$ (both observed and future)
- Channel: $\{(j,t) : j \neq i\}$ (other channels at same future timestep)

**(c)** After each iteration, **reset observed Z**:

$$
Q^{(\tau+1, \text{dec})}_{i,t} \leftarrow \mathbf{Z}^{\text{enc}}_{i,t}, \quad \forall\, t \in \{1, \ldots, P\}
$$

**(d)** Ternary, binary factors are **shared** with the encoder.

**(e)** RoPE positions naturally extend to $\{0, 1, \ldots, P+S_p-1\}$.

**Output:** $\mathbf{Z}^{\text{dec}} = \{Q^{(K_{\text{dec}})}_{i,t}\}_{t \in \{P+1,\ldots,P+S_p\}}$, shape $[B, N, S_p, d]$.

> **Physical intuition:** Each decoder MFVI iteration propagates information one "hop" from $\mathbf{Z}^{\text{enc}}$ to future Z through ternary factors. After $K_{\text{dec}}$ iterations, information has propagated $K_{\text{dec}}$ hops. All future Z are updated **in parallel** — **no sequential dependency, no accumulative error**.

---

## 5. Training Objective

### 5.1 Oracle Target Generation (Training Only)

At training time, we have access to ground truth $Y$. Construct full unary factors for all $P + S_p$ positions and run standard Encoder MFVI:

$$
\mathbf{Z}^{\text{target}} = \text{sg}\!\left[\text{MFVI}\!\left(\left[S^{\text{obs}}_{1:P},\; \text{MLP}_{\text{unary}}(\mathbf{y}_{P+1:P+S_p})\right]\right)\right]
$$

where $\mathbf{y}_{i,t}$ is the GT future patch. **Stop-gradient** is applied to prevent representational collapse.

### 5.2 Latent Consistency Loss (Self-Supervised)

**Primary term** — SmoothL1 on unnormalized Z scores (future positions only):

$$
\mathcal{L}_{\text{latent}} = \frac{1}{N \cdot S_p} \sum_{i=1}^{N} \sum_{t=P+1}^{P+S_p} \text{SmoothL1}\!\left(\text{sg}\!\left[\mathbf{Z}^{\text{target}}_{i,t}\right],\; \mathbf{Z}^{\text{dec}}_{i,t}\right)
$$

**Optional KL term** — on normalized distributions:

$$
\mathcal{L}_{\text{KL}} = \frac{1}{N \cdot S_p} \sum_{i=1}^{N} \sum_{t=P+1}^{P+S_p} D_{\text{KL}}\!\left(\text{sg}\!\left[\psi(\mathbf{Z}^{\text{target}}_{i,t})\right]\; \big\|\; \psi(\mathbf{Z}^{\text{dec}}_{i,t})\right)
$$

where $\psi$ is the SquaredSoftmax normalization.

### 5.3 Prediction Loss

$$
\hat{Y} = \text{PredHead}\!\left(\mathbf{Z}^{\text{dec}}\right), \quad \mathcal{L}_{\text{pred}} = \text{MSE}(\hat{Y},\; Y)
$$

### 5.4 Total Loss

$$
\boxed{\mathcal{L} = \mathcal{L}_{\text{pred}} + \lambda_{\text{latent}} \cdot \mathcal{L}_{\text{latent}} + \lambda_{\text{KL}} \cdot \mathcal{L}_{\text{KL}}}
$$

---

## 6. Complete Training and Inference Procedure

### Training (per batch)

```
Input: (X_{1:T}, Y_{T+1:T+S})

1. Patching + Unary:
   S_obs     = MLP_unary(X_patches)                   # [B, N, P, d]
   S_obs_gt  = MLP_unary([X_patches, Y_patches])      # [B, N, P+S_p, d]

2. Encoder MFVI (K_enc iterations, observed region):
   Z_enc ← MFVI(S_obs)                                # [B, N, P, d]

3. Decoder MFVI (K_dec iterations, fix Z_enc):
   Z_dec ← DecoderMFVI(Z_enc, S_prior)                # [B, N, S_p, d]

4. Oracle MFVI (K_enc iterations, full region, stop-grad):
   Z_target ← sg[MFVI(S_obs_gt)]                      # [B, N, P+S_p, d]

5. Loss:
   L_pred    = MSE(PredHead(Z_dec), Y)
   L_latent  = SmoothL1(Z_target[P+1:], Z_dec)
   L_KL      = KL(ψ(Z_target[P+1:]) || ψ(Z_dec))
   L         = L_pred + λ_latent · L_latent + λ_KL · L_KL
```

### Inference

```
Input: X_{1:T}

1. Encoder MFVI → Z_enc
2. Decoder MFVI → Z_dec
3. PredHead(Z_dec) → Ŷ

No oracle. No extra model. No accumulative error.
```

---

## 7. Key Design Summary

| Design Choice | Justification | Code Location |
|---|---|---|
| Encoder-Decoder separation | Observed Z not affected by future Z | New decoder phase |
| Fix Z_enc during decoder | Preserves belief-state causality | Reset after each decoder iteration |
| Shared ternary factors | Same CRF dynamics everywhere | `ternary_factor_u/v_time/channel` |
| Joint H normalization | Temporal / channel dependency competition | `combined_qh` softmax |
| Learnable prior $S^{\text{prior}}$ | Replaces unary for future Z | New parameter |
| Stop-gradient on oracle | Prevents representational collapse | `sg[Z_target]` |
| SmoothL1 + KL dual loss | Aligns scores and distributions | Following NextLat |
| RoPE extension to $P+S_p$ | Encodes observed-future relative distance | `position_ids` extended |

---

## 8. Differentiation from NextLat

| Dimension | NextLat | Our Approach |
|---|---|---|
| **Prediction paradigm** | Autoregressive state transition | Posterior inference on partially-observed PGM |
| **Dynamics source** | External MLP $p_\psi$ | CRF's own ternary factors via message passing |
| **Self-supervised target** | Sees next observation $X_{t+1}$ as input | Infers future Z **without** future observations |
| **Multi-step prediction** | Sequential rollout (accumulative error) | Parallel MFVI (no accumulative error) |
| **Training = Inference?** | No ($p_\psi$ only at training) | Yes (Decoder MFVI used in both) |
| **Extra parameters** | $p_\psi$ MLP | None |
| **Latent semantics** | Arbitrary continuous vectors | Probability distributions with CRF semantics |
