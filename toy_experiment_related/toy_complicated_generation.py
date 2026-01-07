import numpy as np
import pandas as pd
import os
import json

def generate_stpt_compatible_dataset(length=10000):
    """
    生成符合 Time-Series-Library 要求的 CSV 数据集。
    包含 ST-PT 论文中讨论的 8 种核心先验场景 (Priors)。
    
    参数:
        length: 序列总长度。建议至少 10000+ 以保证 Train/Test 都有足够完整的模式。
                TSLib 默认按 70% 训练, 10% 验证, 20% 测试划分。
    """
    np.random.seed(2024)
    # TSLib 需要连续的时间点，这里模拟从 2020年开始的小时数据
    date_range = pd.date_range(start='2020-01-01', periods=length, freq='h')
    
    # 为了保证波形的密度和原来的类似，我们需要调整时间变量 t 的跨度
    # 原来 2000 点走了 100个单位，现在 10000 点走 500 个单位
    t = np.linspace(0, 500, length) 
    
    data = {}
    
    # -------------------------------------------------------------
    # 1. Periodicity (周期性) - Dim 0
    # -------------------------------------------------------------
    # 调整 period: 如果采样率变高，period 也要相应变大以保持波形视觉一致
    # 假设这里的 period 对应 24 小时
    period_points = 24 
    angular_freq = 2 * np.pi / (500/length * period_points) # 计算角频率
    
    signal_periodic = np.sin(t * angular_freq) 
    signal_periodic += 0.5 * np.cos(t * angular_freq * 2) # 加入二倍频谐波
    data['dim0'] = signal_periodic + np.random.normal(0, 0.2, length)

    # -------------------------------------------------------------
    # 2. Trend (趋势) - Dim 1
    # -------------------------------------------------------------
    trend = np.linspace(0, 20, length) # 扩大趋势幅度
    data['dim1'] = trend + np.sin(t) * 0.5 + np.random.normal(0, 0.1, length)

    # -------------------------------------------------------------
    # 3. Grouping (分组共性) - Dim 2 & Dim 3
    # -------------------------------------------------------------
    shared_latent = np.cos(t * 0.5) 
    data['dim2'] = shared_latent + np.random.normal(0, 0.3, length)
    data['dim3'] = -1.0 * shared_latent + np.random.normal(0, 0.3, length)

    # -------------------------------------------------------------
    # 4. Local Smoothness (平滑/惯性) - Dim 4
    # -------------------------------------------------------------
    smooth_rw = np.zeros(length)
    for i in range(1, length):
        # 加上均值回归项防止游走到无穷大 (对于长序列很重要)
        smooth_rw[i] = 0.99 * smooth_rw[i-1] + np.random.normal(0, 0.1)
    data['dim4'] = smooth_rw

    # -------------------------------------------------------------
    # 5. Lagged Effect (因果滞后) - Dim 5 & Dim 6
    # -------------------------------------------------------------
    source_sig = np.sign(np.sin(t * 0.2)) # 方波
    data['dim5'] = source_sig + np.random.normal(0, 0.1, length)
    
    tau = 24 # 滞后 24 步
    target_sig = np.roll(source_sig, shift=tau)
    target_sig[:tau] = 0
    data['dim6'] = target_sig + np.random.normal(0, 0.5, length)

    # -------------------------------------------------------------
    # 6. Independence (纯噪声) - Dim 7
    # -------------------------------------------------------------
    data['dim7'] = np.random.normal(0, 1.0, length)

    # -------------------------------------------------------------
    # 7. Regime Shift (分布突变) - Dim 8
    # -------------------------------------------------------------
    # 在 70% 处发生突变（正好在 Train 结束进入 Test 时，非常有挑战性）
    shift_point = int(length * 0.7)
    regime = np.zeros(length)
    regime[:shift_point] = np.sin(t[:shift_point] * 5)
    regime[shift_point:] = 2.0 
    data['OT'] = regime + np.random.normal(0, 0.1, length)

    # # -------------------------------------------------------------
    # # 8. Events (稀疏脉冲) - Dim 9
    # # -------------------------------------------------------------
    # events = np.zeros(length)
    # base = np.random.normal(0, 0.1, length)
    # num_events = length // 100
    # event_indices = np.random.choice(range(length), size=num_events, replace=False)
    # # 转换为 int 以支持 json 序列化
    # event_indices = [int(x) for x in event_indices]
    # events[event_indices] = 5.0
    # data['dim9'] = base + events

    # -------------------------------------------------------------
    # 构造 DataFrame
    # -------------------------------------------------------------
    df = pd.DataFrame(data)
    
    # 关键：加上 date 列
    df.insert(0, 'date', date_range)

    # return df, event_indices
    return df

if __name__ == "__main__":
    save_dir = './dataset/toy_data/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Generating ST-PT scenario dataset...")
    # df, evt_idx = generate_stpt_compatible_dataset(length=12000)
    df = generate_stpt_compatible_dataset(length=1200)
    # 保存 CSV
    save_path = os.path.join(save_dir, 'toy_complicated.csv')
    df.to_csv(save_path, index=False)
    print(f"Dataset Saved: {save_path}")
    print(f"Shape: {df.shape}")
    print("Columns:", df.columns.tolist())
    
    # # 选择性保存 metadata (如果你需要用来验证 Prior Mask)
    # metadata = {
    #     "periodicity": {"dims": ["dim0"], "period": 24},
    #     "trend": {"dims": ["dim1"]},
    #     "grouping": {"groups": [["dim2", "dim3"]]},
    #     "smoothness": {"dims": ["dim4"]},
    #     "causal": {"pairs": [["dim5", "dim6"]], "lag": 24},
    #     "independence": {"dims": ["dim7"]},
    #     "locality": {"dims": ["dim8"]},
    #     "events": {"dims": ["dim9"], "indices": evt_idx}
    # }
    # with open(os.path.join(save_dir, 'toy_stpt_metadata.json'), 'w') as f:
    #     json.dump(metadata, f, indent=4)
    #     print("Metadata saved to json.")