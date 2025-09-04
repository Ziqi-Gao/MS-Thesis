#!/usr/bin/env python3
# 文件名: normality_check.py
# 功能：对 Concept.npz 中每层每个维度的 w_all 进行正态性检验（Shapiro-Wilk 与 D’Agostino’s K²），
# 分别生成两个 18×2048 的 "pass"/"fail" 表格，并统计每层的通过/失败个数及通过率。

import numpy as np
from scipy.stats import shapiro, normaltest
import pandas as pd


def main():
    # 1. 载入数据
    data = np.load('/gpfs/projects/p32737/del6500_home/CD/lab_rs/google-gemma-2b_STSA_w.npz')['W']
    # data.shape == (n_iter, n_layers, hidden_size)
    n_iter, n_layers, hidden_size = data.shape

    # 2. 初始化两个标签表：Shapiro 和 D’Agostino
    shapiro_table = np.empty((n_layers, hidden_size), dtype=object)
    dagostino_table = np.empty((n_layers, hidden_size), dtype=object)
    alpha = 0.05  # 显著性水平

    # 3. 按层按维度检验
    for layer in range(n_layers):
        for dim in range(hidden_size):
            vals = data[:, layer, dim]
            # Shapiro–Wilk 检验
            _, p_sw = shapiro(vals)
            shapiro_table[layer, dim] = 'pass' if p_sw > alpha else 'fail'
            # D’Agostino’s K² 检验
            _, p_dk2 = normaltest(vals)
            dagostino_table[layer, dim] = 'pass' if p_dk2 > alpha else 'fail'

    # 4. 保存标签表为 CSV
    index = [f'layer_{i}' for i in range(n_layers)]
    columns = [f'dim_{j}' for j in range(hidden_size)]
    df_sw = pd.DataFrame(shapiro_table, index=index, columns=columns)
    df_dk2 = pd.DataFrame(dagostino_table, index=index, columns=columns)
    out_sw = './shapiro_table.csv'
    out_dk2 = './dagostino_table.csv'
    df_sw.to_csv(out_sw)
    df_dk2.to_csv(out_dk2)
    print(f"Shapiro table saved to {out_sw}")
    print(f"D'Agostino table saved to {out_dk2}")

    # 5. 统计每层的通过/失败数量及通过率
    summary = []
    for layer in range(n_layers):
        sw_pass = np.count_nonzero(shapiro_table[layer, :] == 'pass')
        sw_fail = hidden_size - sw_pass
        dk2_pass = np.count_nonzero(dagostino_table[layer, :] == 'pass')
        dk2_fail = hidden_size - dk2_pass
        summary.append({
            'layer': f'layer_{layer}',
            'shapiro_pass': sw_pass,
            'shapiro_fail': sw_fail,
            'shapiro_rate': sw_pass / hidden_size,
            'dagostino_pass': dk2_pass,
            'dagostino_fail': dk2_fail,
            'dagostino_rate': dk2_pass / hidden_size
        })
    df_summary = pd.DataFrame(summary).set_index('layer')
    out_summary = './normality_summary.csv'
    df_summary.to_csv(out_summary)
    print(f"Summary saved to {out_summary}")

    # 6. 输出统计矩阵
    print(df_summary)


if __name__ == '__main__':
    main()
