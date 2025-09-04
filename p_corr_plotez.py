#!/usr/bin/env python3
"""
batch_corr_overview.py

功能：批量处理指定目录下所有 *_w.npz 文件，
对每个文件中的 W 张量计算每层平均绝对相关度（mean_abs_layer），
并生成相应的折线图，保存到指定输出目录。

用法示例：
  python /gpfs/projects/p32737/del6500_home/CD/p_corr_plotez.py \
    --input_dir /gpfs/projects/p32737/del6500_home/CD/lab_rs \
    --output_dir /gpfs/projects/p32737/del6500_home/CD/plot
"""
import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt

def compute_mean_abs_layer(W: np.ndarray) -> np.ndarray:
    """
    计算 W 张量中每层的平均绝对相关度。
    W 的形状为 (n_iter, num_layers, dim)。
    返回长度为 num_layers 的一维数组。
    """
    # W: shape (n_iter, num_layers, dim)
    n_iter, num_layers, dim = W.shape
    mean_abs_layer = np.zeros(num_layers)
    for l in range(num_layers):
        # 提取第 l 层矩阵：shape (n_iter, dim)
        Wl = W[:, l, :]
        # 计算维度间两两相关系数矩阵
        R = np.corrcoef(Wl, rowvar=False)
        # NaN->0，并去掉对角线
        R = np.nan_to_num(R)
        np.fill_diagonal(R, 0)
        # 平均绝对相关度
        mean_abs_layer[l] = np.mean(np.abs(R))
    return mean_abs_layer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Batch compute and plot mean abs correlation per layer.'
    )
    parser.add_argument(
        '--input_dir', '-i', required=True,
        help='目录路径，包含多个 *_w.npz 文件'
    )
    parser.add_argument(
        '--output_dir', '-o', required=True,
        help='保存输出图的目录'
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # 扫描所有以 _w.npz 结尾的文件
    pattern = os.path.join(input_dir, '*_w.npz')
    file_list = sorted(glob.glob(pattern))
    if not file_list:
        print(f'No files found in {input_dir} matching * _w.npz')
        exit(1)

    for path in file_list:
        fname = os.path.basename(path)
        # 解析文件名：model_dataset_ntrain_w.npz
        stem = fname[:-4]  # 去掉 .npz
        if not stem.endswith('_w'):
            print(f'Skipping unexpected file: {fname}')
            continue
        base = stem[:-2]
        parts = base.rsplit('_', 2)
        if len(parts) != 3:
            print(f'Cannot parse filename: {fname}')
            continue
        model_tag, dataset, n_train = parts

        # 加载 W 张量
        data = np.load(path)
        if 'W' in data:
            W = data['W']
        elif 'w_all' in data:
            W = data['w_all']
        else:
            print(f'Missing W key in {fname}')
            continue

        # 计算平均绝对相关度
        mean_abs = compute_mean_abs_layer(W)

        # 绘制折线图
        plt.figure(figsize=(8, 4))
        plt.plot(range(len(mean_abs)), mean_abs, marker='o')
        plt.xlabel('Layer Index')
        plt.ylabel('Mean Absolute Correlation')
        plt.title(f'{model_tag} | {dataset} | n_train={n_train}')
        plt.grid(True)
        plt.tight_layout()

        # 保存图像
        out_fname = f'{model_tag}_{dataset}_{n_train}_mean_abs_corr.png'
        out_path = os.path.join(output_dir, out_fname)
        plt.savefig(out_path, dpi=150)
        plt.close()

        print(f'Saved overview plot: {out_path}')
