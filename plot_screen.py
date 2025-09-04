#!/usr/bin/env python3
# plot_scree.py
# 绘制每个层的 Scree-plot（累计解释方差）

import os, glob, re, numpy as np
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")                    # 后端设为无窗口
import matplotlib.pyplot as plt

# ---------- 路径与文件模式 ----------
OBS_DIR  = "./lab_rs/obs"                # ← 按需要调整
PLOT_DIR = "./plot/screen"                      # 输出目录
PATTERN  = re.compile(r"(.+)_layer(\d+)_(diag|full)_obs\.npy$")

# ---------- 工具函数（沿用原脚本） ----------
def collect_layer_files():
    """返回 dict{prefix:{layer:int -> filepath}}（diag 优先）"""
    mapping = {}
    for fp in glob.glob(os.path.join(OBS_DIR, "*_obs.npy")):
        m = PATTERN.search(os.path.basename(fp))
        if not m:
            continue
        prefix, layer, tag = m.group(1), int(m.group(2)), m.group(3)
        mapping.setdefault(prefix, {})
        if tag == "diag" or layer not in mapping[prefix]:
            mapping[prefix][layer] = fp
    return mapping

# ---------- 核心：绘制 Scree-plot ----------
def draw_scree(ax, ratios, k70, k90, title):
    cum = np.cumsum(ratios)
    ax.plot(np.arange(1, len(cum) + 1), cum, lw=1.5)
    ax.axvline(k70, color="g", ls="--", alpha=.6, label=f"70% @ {k70}")
    ax.axvline(k90, color="r", ls="--", alpha=.6, label=f"90% @ {k90}")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_ylim(0, 1.01)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="lower right")

def scree_for_layer(npfile, save_path):
    X = np.load(npfile)[:, :-1]          # 去掉最后一列 b
    pca = PCA(svd_solver="full").fit(X)
    ratios = pca.explained_variance_ratio_
    cum    = np.cumsum(ratios)
    k70 = int(np.searchsorted(cum, 0.70) + 1)
    k90 = int(np.searchsorted(cum, 0.90) + 1)

    fig, ax = plt.subplots(figsize=(6, 4))
    title = os.path.basename(save_path).replace("_scree.png", "")
    draw_scree(ax, ratios, k70, k90, title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

# ---------- 主程序 ----------
def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    mapping = collect_layer_files()
    if not mapping:
        print(f"[!] 未找到 *_diag_obs.npy / *_full_obs.npy 文件，请检查 OBS_DIR={OBS_DIR}")
        return

    for prefix, layers in sorted(mapping.items()):
        for layer, fp in sorted(layers.items()):
            out_name = f"{prefix}_layer{layer:02d}_scree.png"
            out_path = os.path.join(PLOT_DIR, out_name)
            print(f"[+] 计算 Scree-plot → {out_path}")
            scree_for_layer(fp, out_path)

if __name__ == "__main__":
    main()
