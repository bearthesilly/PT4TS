# 重新设计三套合成数据实验（Channel Grouping / Sparse Topology / Temporal Decay）

## 核心分析：为什么原来的三套实验失败

### 成功实验的共同模式
观察三套成功的实验（Periodicity / Trend / Lag），它们的 prior 注入机制都是 **添加新的计算路径**：

| 成功实验 | Prior 注入方式 | 关键 |
|---------|--------------|------|
| Periodicity | `messageF = messageF * cos_prior` + `qh = qh * cos_prior` | 乘性调制 F 和 G |
| Trend | 新增 `prediction_trend(qm)` 解码头 + 新增 HMM 隐变量 qm | 新增变量 + 新解码路径 |
| Lag | 新增 `PtLagPrior` 模块产生 `m_lag` 消息 | **新增消息项** → `qz = (m_t + m_c + m_g + m_lag + unary)` |

### 失败实验的共同问题
三套失败实验都只是 **约束现有路径**（用 `-inf` 硬掩码阻断 channel 注意力，或用加性衰减偏置），而非添加新路径。约束型 prior 有两个致命缺陷：
1. 硬 `-inf` 掩码破坏 joint softmax 的时间-通道竞争平衡
2. 约束只减少信息流，不主动提供先验知识

### 设计原则
**每个 prior 必须为模型创建一条新的计算路径（新消息），将先验知识直接编码为可用信号，而不是仅仅约束注意力。** 这与成功的 Lag 实验完全一致：添加 `m_xxx` 到迭代更新中。

---

## 实验 4：Channel Grouping —— 重新设计

### 数据设计："组内去噪"
- **9 通道，3 组 × 3**：Group A {0,1,2}, Group B {3,4,5}, Group C {6,7,8}
- 每组共享一个 **平滑可预测的隐因子**（正弦波 + 慢漂移），但三组的隐因子 **完全不同**
- 每通道 = 组隐因子 + **非常重的独立噪声**（σ=1.5）
- 单通道 SNR ≈ 0.67（几乎不可能仅从单通道预测）
- 组内 3 通道平均 SNR ≈ 0.67 × √3 ≈ 1.16（可以做出合理预测）
- 跨组平均会引入 **错误的隐因子**，使预测更差

**关键设计点**：
- 信号幅度 = 1.0（正弦波），噪声 σ = 1.5
- 三组使用不同周期 (24, 36, 16) 和独立随机相位
- 150 个样本，vanilla 模型难以从嘈杂数据中可靠发现分组结构

### 模型设计：`PtGroupPooling` —— 新消息模块

```python
class PtGroupPooling(nn.Module):
    """Channel Grouping Prior: explicit within-group consensus message."""
    def __init__(self, groups, dim_z):
        # groups = [(0,1,2), (3,4,5), (6,7,8)]
        # 为每组创建一个可学习的投影矩阵 W_g
        self.groups = groups
        self.W = nn.ParameterList([
            nn.Parameter(torch.empty(dim_z, dim_z)) for _ in groups
        ])

    def forward(self, qz_norm):
        # qz_norm: [bs, enc_in, patch_num, dim_z]
        msg = torch.zeros_like(qz_norm)
        for g_idx, group in enumerate(self.groups):
            # 计算组内平均：直接平均同组通道的 Z 表示
            group_avg = qz_norm[:, group].mean(dim=1)  # [bs, patch_num, dim_z]
            # 通过可学习投影变换
            projected = group_avg @ self.W[g_idx]  # [bs, patch_num, dim_z]
            # 发送给组内每个通道
            for ch in group:
                msg[:, ch] = projected
        return msg
```

**注入方式**（与 Lag 完全一致）：
```python
# PtEncoderIterator.forward:
m_group = self.group_pooling(qz)  # 新消息
qz = (m_t + m_c + m_g + m_group + unary_potentials) / regularize_z
```

**为什么有效**：
- Vanilla 模型：必须通过 channel-axis ternary factor 从噪声数据中学习分组→搜索空间大
- Prior 模型：`m_group` 直接提供组内去噪后的共识信号→每次迭代都有干净的组内信息

### 文件变更
- 新建 `toy_experiment_related/syn_channel_group_generation_v2.py`
- 新建 `models/PT_syn_channel_group_v2.py`（基于 PT_forecast_v15 + PtGroupPooling）
- 新建 `scripts/toy_experiment/syn_channel_group_v2.sh`

---

## 实验 5：Sparse Topology —— 重新设计

### 数据设计："邻居协同去噪"
- **8 通道，链式图**：0—1—2—3—4—5—6—7
- 每条边 (i, i+1) 有一个共享的 **平滑可预测的边隐因子**
- 每个通道的信号 = 它关联的边隐因子的平均 + **重噪声** (σ=1.0)
  - ch0 = edge_01 + noise
  - ch1 = (edge_01 + edge_12) / 2 + noise
  - ch7 = edge_67 + noise
- 信号幅度 ≈ 1.0，噪声 σ = 1.0，SNR ≈ 1.0（单通道勉强可预测）
- 但如果利用邻居通道的信息协同去噪，可以更好地恢复边隐因子→更好预测

**关键设计点**：
- 邻居共享边隐因子 → 邻居信息直接有用
- 非邻居不共享任何边隐因子 → 非邻居信息是纯噪声
- 150 样本下，vanilla 模型难以精确学到哪些通道是邻居

### 模型设计：`PtTopologyMessage` —— 新消息模块

```python
class PtTopologyMessage(nn.Module):
    """Sparse Topology Prior: explicit neighbor-aggregation message."""
    def __init__(self, adjacency, dim_z):
        # adjacency = [(0,1), (1,2), ..., (6,7)]
        self.adjacency = adjacency
        # 为每条边创建一个可学习的投影矩阵
        self.edge_W = nn.ParameterList([
            nn.Parameter(torch.empty(dim_z, dim_z)) for _ in adjacency
        ])

    def forward(self, qz_norm):
        # qz_norm: [bs, enc_in, patch_num, dim_z]
        msg = torch.zeros_like(qz_norm)
        for e_idx, (i, j) in enumerate(self.adjacency):
            W = self.edge_W[e_idx]
            # 双向消息：i→j 和 j→i
            msg[:, j] = msg[:, j] + qz_norm[:, i] @ W
            msg[:, i] = msg[:, i] + qz_norm[:, j] @ W.T
        return msg
```

**注入方式**：
```python
# PtEncoderIterator.forward:
m_topo = self.topology_message(qz)
qz = (m_t + m_c + m_g + m_topo + unary_potentials) / regularize_z
```

**为什么有效**：
- Vanilla 模型：channel-axis attention 对所有通道对等处理，在少样本下难以从噪声中辨别邻接关系
- Prior 模型：`m_topo` 只沿已知边传递消息→直接利用邻居的共享边隐因子信息

### 文件变更
- 新建 `toy_experiment_related/syn_sparse_topology_generation_v2.py`
- 新建 `models/PT_syn_sparse_topo_v2.py`（基于 PT_forecast_v15 + PtTopologyMessage）
- 新建 `scripts/toy_experiment/syn_sparse_topology_v2.sh`

---

## 实验 6：Temporal Decay —— 重新设计

### 数据设计："变点检测"
- **10 通道**，每个样本在输入窗口 (96 步) 的 **随机位置 τ ∈ [20, 76]** 发生一次突变
- 突变前：所有通道遵循 **模式 A**（特定频率/斜率）
- 突变后：所有通道切换到 **模式 B**（完全不同的频率/斜率）
- **预测窗口** (96 步) 始终延续 **模式 B**
- 添加中等噪声 (σ=0.2)

通道设计：
- Group A (ch0-3): 正弦波，突变前周期 P₁=48，突变后周期 P₂=12（频率差 4 倍）
- Group B (ch4-6): 线性趋势，突变前斜率 +0.1，突变后斜率 -0.1（方向反转）
- Group C (ch7-9): **无突变的平稳周期信号**（对照组，验证 decay 不伤害平稳信号）

**关键设计点**：
- 突变位置 τ 每个样本随机 → 模型不能学到固定位置
- 模式 A 和 B 差异巨大 → 用错误模式预测会产生大误差
- Vanilla 模型对所有时间步等权处理 → 混合了新旧模式 → 预测偏差
- 带 decay prior 的模型聚焦最近的时间步 → 只看模式 B → 预测准确

### 模型设计：Decay-weighted unary + F/G 乘性调制

**双重注入**：

1. **Unary 层面**：对初始 unary potential 施加衰减权重
```python
# 在 PtModel.forward 中，生成 unary_potentials 后：
# weights[i] = exp(-alpha * (patch_num - 1 - i))，最近的 patch 权重 = 1
decay_weights = torch.exp(-alpha * torch.arange(patch_num-1, -1, -1))  # [patch_num]
unary_potentials = unary_potentials * decay_weights.view(1, 1, patch_num, 1)
```

2. **MessageF/G 层面**：乘性 exp 衰减（对称注入）
```python
# decay_prior[i,j] = exp(-alpha * |i - j|)，值域 (0, 1]
# calculate_messageF (time): message_F = message_F * decay_prior
# calculate_messageG (time): qh = qh * decay_prior
```

**为什么有效**：
- Unary 衰减：直接降低旧时间步的初始信号强度→旧模式的信息从源头被削弱
- F/G 乘性衰减：时间轴注意力偏好近邻 patch→减少来自旧模式的消息
- 双重作用叠加，对变点数据效果显著

### 文件变更
- 新建 `toy_experiment_related/syn_temporal_decay_generation_v2.py`
- 新建 `models/PT_syn_temporal_decay_v2.py`（基于 PT_forecast_v15 + 双重 decay 注入）
- 新建 `scripts/toy_experiment/syn_temporal_decay_v2.sh`

---

## 实施清单

### 新文件（共 9 个）
1. `toy_experiment_related/syn_channel_group_generation_v2.py`
2. `toy_experiment_related/syn_sparse_topology_generation_v2.py`
3. `toy_experiment_related/syn_temporal_decay_generation_v2.py`
4. `models/PT_syn_channel_group_v2.py`
5. `models/PT_syn_sparse_topo_v2.py`
6. `models/PT_syn_temporal_decay_v2.py`
7. `scripts/toy_experiment/syn_channel_group_v2.sh`
8. `scripts/toy_experiment/syn_sparse_topology_v2.sh`
9. `scripts/toy_experiment/syn_temporal_decay_v2.sh`

### 需要修改的文件
10. `models/__init__.py` 或 `exp/exp_basic.py` — 注册新模型名

### 不修改的文件
- 原有的 v1 版本保持不变（对比用）
- `data_provider/data_loader.py` — Toy_Dataset 类已支持 npy 格式，无需修改
- `run.py` — 参数系统已足够

### 每套实验的对比模型（与原有实验一致）
- PT_syn_xxx_v2（带 prior 的新模型）
- PT_forecast_v15（vanilla ST-PT）
- DLinear
- BVAR
