import sys
import os

# 将当前目录加入到 Python 的搜索路径中
sys.path.append(os.getcwd())

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
# 假设你已经把 Toy_Dataset 代码粘贴到了 data_provider/data_loader.py 中
# 根据你的文件结构，可能需要调整 import 路径
from data_provider.data_loader import Toy_Dataset

def test_toy_dataset_workflow():
    # --- 1. 配置参数 ---
    root_path = './dataset/toy_data/'
    data_filename = 'test_iid_data.npy'
    
    # 设定数据维度
    N_samples = 100   # 样本数量
    T_length = 200    # 每个样本的时间步长 (必须 > seq_len + pred_len)
    C_channels = 8    # 变量数
    
    # 设定窗口参数 (模拟 run.sh 中的参数)
    seq_len = 96
    label_len = 48
    pred_len = 24
    
    if not os.path.exists(root_path):
        os.makedirs(root_path)
        
    # --- 2. 生成假数据 (N, T, C) ---
    print(f"Generating dummy data: Shape ({N_samples}, {T_length}, {C_channels})...")
    # 生成一些带有特定 pattern 的数据，方便 debug (例如全 1)
    # 或者直接用随机数
    dummy_data = np.random.randn(N_samples, T_length, C_channels).astype(np.float32)
    
    save_path = os.path.join(root_path, data_filename)
    np.save(save_path, dummy_data)
    print(f"Data saved to {save_path}")
    
    # --- 3. 实例化 Toy_Dataset ---
    print("\nInitializing Toy_Dataset...")
    try:
        dataset = Toy_Dataset(
            root_path=root_path,
            flag='train',          # 测试训练集模式
            data_path=data_filename,
            size=[seq_len, label_len, pred_len],
            features='M',          # 多变量预测
            scale=True             # 测试这一步是否报错
        )
    except Exception as e:
        print(f"!! Initialization Failed: {e}")
        return

    print(f"Dataset Length (should be roughly 70% of {N_samples}): {len(dataset)}")

    # --- 4. 模拟读取一个 Batch ---
    # 使用 DataLoader 模拟真实的训练过程
    batch_size = 16
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    print("\nFetching one batch from DataLoader...")
    try:
        # 尝试获取一个batch
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):
            
            # --- 5. 形状检查 (Assertion) ---
            print(f"Batch {i} Shapes:")
            print(f"  batch_x:      {batch_x.shape}")
            print(f"  batch_y:      {batch_y.shape}")
            print(f"  batch_x_mark: {batch_x_mark.shape}")
            print(f"  batch_y_mark: {batch_y_mark.shape}")
            
            # 检查 seq_x 形状: [Batch, seq_len, Channels]
            expected_x_shape = (batch_size, seq_len, C_channels)
            assert batch_x.shape == expected_x_shape, \
                f"batch_x shape mismatch! Expected {expected_x_shape}, got {batch_x.shape}"
            
            # 检查 seq_y 形状: [Batch, label_len + pred_len, Channels]
            expected_y_shape = (batch_size, label_len + pred_len, C_channels)
            assert batch_y.shape == expected_y_shape, \
                f"batch_y shape mismatch! Expected {expected_y_shape}, got {batch_y.shape}"
            
            # 检查 mark 形状 (时间特征通常是 4 维: month/day/weekday/hour)
            # 或者 TSLib 默认使用特定编码维度，此处主要检查时间步长是否对齐
            assert batch_x_mark.shape[1] == seq_len, \
                f"batch_x_mark length mismatch! Expected {seq_len}, got {batch_x_mark.shape[1]}"
            
            assert batch_y_mark.shape[1] == label_len + pred_len, \
                f"batch_y_mark length mismatch! Expected {label_len + pred_len}, got {batch_y_mark.shape[1]}"
            
            print("\n✅ SUCCESS: All shapes match TSLib requirements!")
            break # 只跑一个 batch 就够了
            
    except Exception as e:
        print(f"!! Batch Fetching Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_toy_dataset_workflow()