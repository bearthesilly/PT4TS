import pandas as pd
import numpy as np
import os

def generate_data(length=10000):
    t = np.linspace(0, 100 * np.pi, length)
    
    # Dim 1: 基础正弦波 + 小随机噪声
    col1 = np.sin(t) + np.random.normal(0, 0.05, length)
    
    # Dim 2: 余弦波 (与 Dim 1 类似但有相位差) + 噪声
    col2 = np.cos(t) + np.random.normal(0, 0.05, length)
    
    # Dim 3: 频率加倍 + 幅度减半 + 与 Dim 1 相加 (模拟非线性耦合)
    col3 = 0.5 * np.sin(2 * t) + 0.5 * col1 + np.random.normal(0, 0.05, length)

    # 组合数据 [Length, 3]
    data = np.stack([col1, col2, col3], axis=1)
    
    # 创建 DataFrame
    df = pd.DataFrame(data, columns=['dim1', 'dim2', 'dim3'])
    
    # 添加必要的 date 列 (每小时一条)
    df.insert(0, 'date', pd.date_range(start='2020-01-01', periods=length, freq='h'))
    
    return df

if __name__ == "__main__":
    save_dir = './dataset/toy_data/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    df = generate_data(length=10000)
    save_path = os.path.join(save_dir, 'toy_sine.csv')
    
    df.to_csv(save_path, index=False)
    print(f"Generated! Saved to: {save_path}")
    print(f"The shape of the data: {df.shape}")
    print(df.head())