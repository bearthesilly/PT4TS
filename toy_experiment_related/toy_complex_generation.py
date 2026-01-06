import pandas as pd
import numpy as np
import os

def generate_complex_data(length=10000):
    np.random.seed(42) # 固定种子以便复现
    t = np.linspace(0, 100 * np.pi, length) # 时间轴
    
    data_dict = {}

    # --- 1. dim0: 多周期叠加 (Multi-Periodicity) ---
    # 就像电力负荷：有日周期，也有年周期
    data_dict['dim0'] = np.sin(t) + 0.5 * np.sin(5 * t) + np.random.normal(0, 0.05, length)

    # --- 2. dim1: 长期趋势 (Trend) ---
    # 就像股票大盘长期的涨势，或者设备的老化累积
    # 为了防止数值过大，加一个 sigmoid 把它限制在一定范围内，或者做一个缓慢的线性增长
    trend = np.linspace(0, 5, length) 
    # 加上一点季节性
    data_dict['dim1'] = trend + 0.3 * np.sin(0.5 * t) + np.random.normal(0, 0.05, length)

    # --- 3. dim2: 惯性 (Inertia / Random Walk) ---
    # 下一步大概率在这一步附近，有很强的惯性，没有均值回归特性
    rw = np.zeros(length)
    rw[0] = 0
    for i in range(1, length):
        # 惯性项：0.99 * 上一步 + 噪声
        rw[i] = 0.995 * rw[i-1] + np.random.normal(0, 0.1)
    data_dict['dim2'] = rw

    # --- 4. dim3: 激烈跳脱 (Volatile / Noise) ---
    # 几乎没有可预测性，纯粹扰动
    data_dict['dim3'] = np.random.normal(0, 1.5, length) # 方差大，跳脱

    # --- 5. dim4: 延迟响应 (Delayed Interaction) ---
    # 它是 dim0 的回声，延迟 20 个时间步
    delay_step = 20
    # 先生成原始信号
    base_sig = data_dict['dim0']
    # 制造延迟：前20步填0，后面是平移过来的
    shifted_sig = np.roll(base_sig, delay_step)
    shifted_sig[:delay_step] = 0
    data_dict['dim4'] = shifted_sig + np.random.normal(0, 0.05, length)

    # --- 6. dim5: 复杂耦合 (Complex Coupling) ---
    # dim1 (Trend) 的当前值 减去 dim2 (Inertia) 10步之前的值
    # 考验模型跨变量、跨时间步的推理能力
    delay_dim2 = 10
    shifted_dim2 = np.roll(data_dict['dim2'], delay_dim2)
    shifted_dim2[:delay_dim2] = 0
    # 为了数据量级统一，做个简单的缩放
    data_dict['dim5'] = (data_dict['dim1'] - shifted_dim2) * 0.5 + np.random.normal(0, 0.05, length)

    # --- 7. dim6: 短视 (Short-term Dependency - MA process) ---
    # 移动平均过程，只和最近几步的噪声有关
    noise = np.random.normal(0, 0.5, length)
    ma_series = np.zeros(length)
    for i in range(5, length):
        # 当前值只取决于过去 5 步的噪声平均，和 100 步之前完全没关系
        ma_series[i] = np.mean(noise[i-5:i]) + noise[i]
    data_dict['dim6'] = ma_series

    # --- 8. dim7: 全局依赖 (Global Dependency / Low Frequency) ---
    # 一个极长波长的正弦波，波长几乎覆盖整个数据集
    # 它的每一个点都暗示了自己在整条历史长河中的绝对位置
    long_t = np.linspace(0, 2 * np.pi, length) # 整个数据全长才走完一个圆周
    data_dict['dim7'] = np.sin(long_t)

    # --- 9. dim8: 逻辑突变 (Regime Shift / Concept Drift) ---
    # 第 5000 步发生事件
    regime_series = np.zeros(length)
    split_point = 5000
    # 前半段：高频正弦波
    regime_series[:split_point] = np.sin(10 * t[:split_point])
    # 后半段：突变为均值为 2 的随机游走
    rw_part = np.zeros(length - split_point)
    temp_val = 2.0
    for i in range(len(rw_part)):
        temp_val += np.random.normal(0, 0.1)
        rw_part[i] = temp_val
    regime_series[split_point:] = rw_part
    data_dict['dim8'] = regime_series

    # --- 10. dim9: 异常脉冲 (Spikes / Outliers) ---
    # 正常是平稳的，不时来一个大跳变
    base_smooth = 0.5 * np.cos(t)
    spikes = np.zeros(length)
    # 每一百步左右，随机给一个大脉冲
    num_spikes = length // 100
    spike_indices = np.random.choice(range(length), num_spikes, replace=False)
    spikes[spike_indices] = np.random.choice([3.0, -3.0], num_spikes) # 极其跳脱的值
    data_dict['dim9'] = base_smooth + spikes + np.random.normal(0, 0.1, length)

    # --- 组合与整理 ---
    df = pd.DataFrame(data_dict)
    
    # 插入 date 列 (必须)
    df.insert(0, 'date', pd.date_range(start='2020-01-01', periods=length, freq='h'))
    
    return df

if __name__ == "__main__":
    save_dir = './dataset/toy_data/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    print("Generating complex toy dataset...")
    df = generate_complex_data(length=10000)
    save_path = os.path.join(save_dir, 'toy_complex.csv')
    
    df.to_csv(save_path, index=False)
    print(f"Dataset Saved: {save_path}")
    print(f"Shape: {df.shape}")
    print(df.head())