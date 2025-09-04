#!/usr/bin/env python3
# check_concept_cov_conditions.py
# 诊断 observed concept vectors (去掉 b 列)，评估完整协方差是否可靠

import os, glob, re, numpy as np
from sklearn.decomposition import PCA
from scipy.stats import kurtosis
from scipy.spatial.distance import cosine
from sklearn.utils.extmath import safe_sparse_dot

OBS_DIR = "./lab_rs/obs"               # ← 按实际路径调整
PATTERN = re.compile(r"(.+)_layer(\d+)_(diag|full)_obs\.npy$")

# ────────────────────────── 核心工具 ──────────────────────────
def safe_corr_triu(X):
    """
    计算相关矩阵上三角向量（排对角）。
    自动剔除零方差列，使用 float32 降内存。
    返回: tri (len=K*(K-1)//2), K=有效维度
    """
    X = X.astype(np.float32, copy=False)
    M, D = X.shape
    X -= X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, ddof=1)
    nonzero = std > 0
    X = X[:, nonzero] / std[nonzero]
    K = X.shape[1]
    if K < 2:
        raise ValueError("有效维度不足 2")
    C = safe_sparse_dot(X.T, X, dense_output=True) / (M - 1)
    iu = np.triu_indices(K, 1)
    return C[iu], K

def bootstrap_stability(X, n_iter=10, seed=0):
    """两折 bootstrap 相关矩阵相似度均值 (cosine)"""
    rng = np.random.default_rng(seed)
    sims = []
    for _ in range(n_iter):
        idx = rng.choice(len(X), size=len(X), replace=True)
        tri1, k1 = safe_corr_triu(X[idx[: len(X) // 2]])
        tri2, k2 = safe_corr_triu(X[idx[len(X) // 2 :]])
        k = min(len(tri1), len(tri2))
        sims.append(1 - cosine(tri1[:k], tri2[:k]))
    return float(np.mean(sims))

def analyze_layer(X, thresh=0.9):
    """返回 (M/D, rank90, mean|rho|, bootSim, medKurt)"""
    M, D = X.shape
    ratio = M / D

    # PCA 低秩性
    pca = PCA(svd_solver="full").fit(X)
    k90 = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), thresh) + 1)

    # 相关矩阵绝对相关
    tri, _ = safe_corr_triu(X)
    mean_abs_rho = float(np.mean(np.abs(tri)))

    # bootstrap 稳定度
    stab = bootstrap_stability(X)

    # 峭度
    med_kurt = float(np.median(kurtosis(X, axis=0, fisher=False)))

    return ratio, k90, mean_abs_rho, stab, med_kurt

# ────────────────────────── 文件收集 ──────────────────────────
def collect_layer_files():
    """
    返回 dict{prefix: {layer:int -> filepath}}
    优先 diag，如不存在则用 full
    """
    mapping = {}
    for fp in glob.glob(os.path.join(OBS_DIR, "*_obs.npy")):
        m = PATTERN.search(os.path.basename(fp))
        if not m:
            continue
        prefix, layer, tag = m.group(1), int(m.group(2)), m.group(3)
        mapping.setdefault(prefix, {})
        # diag 优先
        if tag == "diag" or layer not in mapping[prefix]:
            mapping[prefix][layer] = fp
    return mapping

# ────────────────────────── 主程序 ──────────────────────────
def main():
    prefix2layers = collect_layer_files()
    if not prefix2layers:
        print("未找到 *_diag_obs.npy / *_full_obs.npy 文件，请检查 OBS_DIR")
        return

    for prefix, layer_map in sorted(prefix2layers.items()):
        print(f"\n=== {prefix} ===")
        for l in sorted(layer_map):
            X = np.load(layer_map[l])
            X = X[:, :-1]          # 去掉最后一列 b 只保留 w
            try:
                ratio, k90, rho, stab, kurt = analyze_layer(X)
                print(f" layer {l:02d} | "
                      f"M/D={ratio:.2f} | "
                      f"rank90={k90} | "
                      f"mean|rho|={rho:.3f} | "
                      f"bootSim={stab:.3f} | "
                      f"medKurt={kurt:.2f}")
            except ValueError as e:
                print(f" layer {l:02d} | 跳过：{e}")

if __name__ == "__main__":
    main()
