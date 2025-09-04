import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def visualize(true_data, pred_data, save_path, channel_name):
    """
    Generates and saves a plot comparing true data and predicted data.

    Args:
        true_data (np.array): The ground truth data for a single channel.
        pred_data (np.array): The predicted data for a single channel.
        save_path (str): The full path to save the plot file.
        channel_name (str or int): The name or index of the channel being plotted.
    """
    plt.figure(figsize=(15, 5))
    plt.plot(true_data, label='GroundTruth')
    plt.plot(pred_data, label='Prediction')
    plt.title(f'Forecast vs. Truth for Channel: {channel_name}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved plot for channel {channel_name} to {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Visualize forecasting results for specified channels.')
    
    # --- 您需要修改这里的默认值 ---
    # 'setting' 对应于您在 results/ 文件夹下的实验结果目录名
    parser.add_argument('--setting', type=str, 
                        default='ETTh1_96_96_Autoformer_ETTh1_ftM_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_Exp_0', 
                        help='The setting name of the experiment')
                        
    # 要可视化的通道列表，以逗号分隔
    parser.add_argument('--channels', type=str, default='0,1,2,3,4,5,6', 
                        help='Comma-separated list of channel indices to visualize')
                        
    # 要可视化的样本索引
    parser.add_argument('--sample_idx', type=int, default=0, 
                        help='The index of the sample to visualize from the batch')
    
    args = parser.parse_args()

    # 定义结果和输出文件夹
    result_path = os.path.join('./results', args.setting)
    output_dir = './additional_visualize_result_forecast'

    # 检查结果文件是否存在
    pred_file = os.path.join(result_path, 'pred.npy')
    true_file = os.path.join(result_path, 'true.npy')

    if not os.path.exists(pred_file) or not os.path.exists(true_file):
        print(f"Error: Cannot find 'pred.npy' or 'true.npy' in {result_path}")
        print("Please make sure the 'setting' argument is correct and the experiment has been run.")
        return

    # 加载数据
    preds = np.load(pred_file)
    trues = np.load(true_file)
    
    # 创建输出文件夹
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 解析通道列表
    try:
        channels_to_visualize = [int(c.strip()) for c in args.channels.split(',')]
    except ValueError:
        print("Error: Invalid format for --channels. Please provide a comma-separated list of integers (e.g., '0,1,5').")
        return

    # 检查样本索引是否有效
    if args.sample_idx >= trues.shape[0]:
        print(f"Error: sample_idx {args.sample_idx} is out of bounds. This dataset has {trues.shape[0]} samples.")
        return

    # 为每个指定的通道生成并保存图像
    for channel_idx in channels_to_visualize:
        if channel_idx >= trues.shape[2]:
            print(f"Warning: Channel index {channel_idx} is out of bounds for this data (total channels: {trues.shape[2]}). Skipping.")
            continue

        true_data = trues[args.sample_idx, :, channel_idx]
        pred_data = preds[args.sample_idx, :, channel_idx]
        
        # 定义输出文件名
        save_path = os.path.join(output_dir, f'{args.setting}_sample{args.sample_idx}_channel{channel_idx}.pdf')
        
        # 可视化
        visualize(true_data, pred_data, save_path, channel_idx)

if __name__ == '__main__':
    main()