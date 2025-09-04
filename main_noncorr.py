#!/usr/bin/env python3
"""
程序 1：对角协方差高斯模型（各维度独立）。
对每一层的 [w|b] 向量拟合对角协方差的高斯分布，
然后从该分布中采样新向量，在留出集上评估准确率，
并通过 PCA 可视化结果。
假设已加载：
W (1000, 18, d)、B (1000, 18)、X_test (n_samples, d)、y_test (n_samples,)
"""
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 为了保证随机采样可复现
np.random.seed(42)

# 加载保存的权重 W 和偏置 B 向量，以及留出测试集数据（根据需要调整路径）
W = np.load('W.npy')   # 形状 (1000, 18, d)
B = np.load('B.npy')   # 形状 (1000, 18)
X_test = np.load('X_test.npy')   # 形状 (n_samples, d)
y_test = np.load('y_test.npy')   # 形状 (n_samples,)

# 帮助函数：给定权重向量 w 和偏置 b，在留出集上计算准确率
def compute_accuracy(w, b, X, y):
    logits = X.dot(w) + b
    # 如果标签为 0/1，则阈值 0 分隔
    if y.max() <= 1 and y.min() >= 0:
        preds = (logits >= 0).astype(int)
    else:
        # 如果标签为 -1/1，则用阈值 0 得到 -1 或 1
        preds = np.where(logits >= 0, 1, -1)
    return np.mean(preds == y)

num_layers = W.shape[1]  # 层数（应为 18）
d = W.shape[2]           # 权重维度 d

for layer in range(num_layers):
    # 为当前层准备扩展后数据 [w|b]
    w_layer = W[:, layer, :]             # 形状 (1000, d)
    b_layer = B[:, layer]                # 形状 (1000,)
    data_layer = np.hstack([w_layer, b_layer.reshape(-1, 1)])  # 形状 (1000, d+1)

    # 在每个维度上估计均值和方差（对角协方差假设）
    mean_vec = data_layer.mean(axis=0)   # 形状 (d+1,)
    var_vec  = data_layer.var(axis=0)    # 形状 (d+1,)
    std_vec  = np.sqrt(var_vec)          # 每维标准差

    # 从 N(mean_vec, diag(var_vec)) 中采样 1000 个新向量 [w|b]
    sampled_data = np.random.randn(1000, d+1) * std_vec + mean_vec  # 形状 (1000, d+1)

    # 计算原始与采样权重在留出集上的准确率
    orig_accs = np.array([
        compute_accuracy(w, b, X_test, y_test)
        for w, b in zip(w_layer, b_layer)
    ])
    samp_accs = np.array([
        compute_accuracy(vec[:-1], vec[-1], X_test, y_test)
        for vec in sampled_data
    ])

    # 输出当前层的准确率对比
    print(f"Layer {layer}: 原始准确率 = {orig_accs.mean():.4f} ± {orig_accs.std():.4f}, "
          f"采样准确率 = {samp_accs.mean():.4f} ± {samp_accs.std():.4f}")

    # PCA 可视化（原始 vs 采样的二维散点图）
    combined_data = np.vstack([data_layer, sampled_data])  # 形状 (2000, d+1)
    pca = PCA(n_components=2)
    combined_2d = pca.fit_transform(combined_data)
    orig_2d = combined_2d[:data_layer.shape[0]]
    samp_2d = combined_2d[data_layer.shape[0]:]

    plt.figure(figsize=(6,6))
    plt.scatter(orig_2d[:,0], orig_2d[:,1], color='blue', alpha=0.6, label='原始')
    plt.scatter(samp_2d[:,0], samp_2d[:,1], color='red',  alpha=0.6, label='采样')
    plt.legend()
    plt.title(f'第 {layer} 层 PCA：原始 vs 采样（对角）')
    plt.xlabel('主成分 1')
    plt.ylabel('主成分 2')
    plt.tight_layout()
    plt.savefig(f'pca_layer{layer}_diag.png')
    plt.close()
