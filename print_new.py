#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
shapiro_pass_rate_batch.py

对 lab_rs 目录下所有 *_w.npz 文件做 Shapiro 正态性检验，
生成每个实验（模型 + 数据集）的一张折线图，
并为每个模型合并对应 9 个数据集的折线图到一张图。

用法示例：
  python print_new.py \
    --input_dir /gpfs/projects/p32737/del6500_home/CD/lab_rs \
    --output_dir /gpfs/projects/p32737/del6500_home/CD/plot \
    --alpha 0.05
"""
import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro


def compute_pass_rate(W: np.ndarray, alpha: float) -> list:
    """
    计算每层通过率：对于 W 的每一层 W[:, l, :],
    单维度做 Shapiro 检验，p >= alpha 视为通过，
    返回长度为 num_layers 的通过率列表（pass_count / dim）。
    """
    n_iter, num_layers, dim = W.shape
    rates = []
    for l in range(num_layers):
        X = W[:, l, :]
        # 对每个维度做单尾 Shapiro 检验
        pass_count = 0
        for d in range(dim):
            _, p = shapiro(X[:, d])
            if p >= alpha:
                pass_count += 1
        rates.append(1 - pass_count / dim)
    return rates


def main():
    parser = argparse.ArgumentParser(
        description="批量对所有实验结果做 Shapiro 检验并生成折线图"
    )
    parser.add_argument(
        "--input_dir", "-i", required=True,
        help="lab_rs 目录，包含多个 *_w.npz 文件"
    )
    parser.add_argument(
        "--output_dir", "-o", required=True,
        help="保存折线图的输出目录"
    )
    parser.add_argument(
        "--alpha", "-a", type=float, default=0.05,
        help="显著性水平 α（默认 0.05），p>=α 视为通过"
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    alpha = args.alpha
    os.makedirs(output_dir, exist_ok=True)

    # 扫描所有 _w.npz 文件
    pattern = os.path.join(input_dir, '*_w.npz')
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No files matching {pattern}")
        return

    # 按模型聚集数据集结果
    model_dict = {}  # model_tag -> list of (dataset, pass_rates)

    for fpath in files:
        fname = os.path.basename(fpath)
        # 文件名示例: model_dataset_ntrain_w.npz
        # 去掉后缀 .npz 及尾部 _w
        stem = fname[:-4]
        if not stem.endswith('_w'):
            continue
        base = stem[:-2]
        parts = base.rsplit('_', 2)
        if len(parts) != 2:
            print(f"Warning: unexpected filename format {fname}")
            continue
        model_tag, dataset = parts #n_train 

        # 加载 W 矩阵
        npz = np.load(fpath)
        if 'W' in npz:
            W = npz['W']
        elif 'w_all' in npz:
            W = npz['w_all']
        else:
            print(f"Warning: no 'W' key in {fname}")
            continue

        # 计算通过率
        pass_rates = compute_pass_rate(W, alpha)

        # # 生成单幅图
        # plt.figure(figsize=(8, 4))
        # plt.plot(range(len(pass_rates)), pass_rates, marker='o')
        # plt.axhline(0.05, color='red', linestyle='--', label='α = 0.05') 
        # plt.xlabel('Layer Index')
        # plt.ylabel('Shapiro Pass Rate')
        # plt.title(f'{model_tag} | {dataset} ')#(n_train={n_train})
        # plt.grid(True)
        # plt.tight_layout()
        # out_file = os.path.join(
        #     output_dir,
        #     f'{model_tag}_{dataset}_passrate.png'#_{n_train}
        # )
        # plt.savefig(out_file, dpi=150)
        # plt.close()
        # print(f"Saved individual plot: {out_file}")

        model_dict.setdefault(model_tag, []).append((dataset, pass_rates))

    # 对每个模型生成合并图
    for model_tag, ds_list in model_dict.items():
        plt.figure(figsize=(10, 6))
        # 按数据集名字排序绘制
        for dataset, rates in sorted(ds_list, key=lambda x: x[0].lower()):
            plt.plot(range(len(rates)), rates, marker='o', label=dataset)
        plt.axhline(0.05, color='red', linestyle='--', label='α = 0.05') 
        plt.xlabel('Layer Index')
        plt.ylabel('Shapiro Pass Rate')
        plt.title(f'{model_tag} | All Datasets')
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        combined_file = os.path.join(
            output_dir,
            f'{model_tag}_combined_pass_rates.png'
        )
        plt.savefig(combined_file, dpi=150)
        plt.close()
        print(f"Saved combined plot: {combined_file}")


if __name__ == '__main__':
    main()
