#!/usr/bin/env python3
# plot_pca_all.py
# 功能：遍历 lab_rs/obs 中所有 “*_layer{l}_diag_obs.npy”，
# 对应地从 lab_rs/samp 加载 diag/full 采样向量，三者一起做 PCA，
# 并将每个模型-数据集-层 的散点图保存到 lab_rs/plot/<model>_<dataset>/。

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 根目录
LAB_RS = "./lab_rs"
OBS_DIR = os.path.join(LAB_RS, "obs")
SAMP_DIR = os.path.join(LAB_RS, "samp")
PLOT_BASE = os.path.join("./plot")

os.makedirs(PLOT_BASE, exist_ok=True)

# 找到所有观测 obs 文件（对角版本），格式：<model>_<dataset>_<n>_layer{l}_diag_obs.npy
obs_paths = glob.glob(os.path.join(OBS_DIR, "*_layer*_diag_obs.npy"))

for obs_path in obs_paths:
    # 解析文件名
    fname = os.path.basename(obs_path)
    # split： [<model>_<dataset>_<n>, 'layer{l}', 'diag_obs.npy']
    parts = fname.split("_layer")
    prefix = parts[0]            # e.g. "google_gemma-2b_cities_149"
    layer_part = parts[1]        # e.g. "0_diag_obs.npy"
    layer_idx = layer_part.split("_")[0]  # e.g. "0"
    
    # 构建对应的 sample 路径
    diag_samp = os.path.join(SAMP_DIR, f"{prefix}_layer{layer_idx}_diag_samp.npy")
    full_samp = os.path.join(SAMP_DIR, f"{prefix}_layer{layer_idx}_full_samp.npy")
    
    # 跳过如果样本不存在
    if not os.path.exists(diag_samp) or not os.path.exists(full_samp):
        print(f"[跳过] 缺少采样文件: {prefix} layer {layer_idx}")
        continue
    
    # 加载数据
    obs_data  = np.load(obs_path)       # (M, d+1)
    diag_data = np.load(diag_samp)      # (M, d+1)
    full_data = np.load(full_samp)      # (M, d+1)
    
    # 合并用于同一 PCA 投影
    combined = np.vstack([obs_data, diag_data, full_data])
    pca = PCA(n_components=2)
    proj = pca.fit_transform(combined)
    
    M = obs_data.shape[0]
    # 切分投影结果
    obs_proj  = proj[            :   M]
    diag_proj = proj[    M   : 2*M]
    full_proj = proj[2*M: 3*M]
    
    # 画图
    plt.figure(figsize=(6,5))
    plt.scatter(obs_proj[:,0],  obs_proj[:,1],  c='blue',  alpha=0.6, label='Observed')
    plt.scatter(diag_proj[:,0], diag_proj[:,1], c='orange',alpha=0.6, label='Gauss-Diag')
    plt.scatter(full_proj[:,0], full_proj[:,1], c='green', alpha=0.6, label='Gauss-Full')
    plt.title(f"PCA Layer {layer_idx}\n{prefix}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    
    # 保存到 lab_rs/plot/<prefix>/
    plot_dir = os.path.join(PLOT_BASE, prefix)
    os.makedirs(plot_dir, exist_ok=True)
    out_path = os.path.join(plot_dir, f"{prefix}_layer{layer_idx}_pca.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Generated PCA plot: {out_path}")
