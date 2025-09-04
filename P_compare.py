#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
# 配置：根据你的情况修改
DATA_DIR = "/gpfs/projects/p32737/del6500_home/CD/lab_rs"         # .npz 文件所在目录
PLOT_DIR = "/gpfs/projects/p32737/del6500_home/CD/plotplot"       # 保存图片的目录
# ──────────────────────────────────────────────────────────────────────────────

def find_result_pairs(data_dir):
    diag_fs = [f for f in os.listdir(data_dir) if f.endswith("_gauss_diag_results.npz")]
    full_fs = set(f for f in os.listdir(data_dir) if f.endswith("_gauss_poet_results.npz"))
    pairs = []
    for diag_fn in diag_fs:
        prefix = diag_fn[:-len("_gauss_diag_results.npz")]
        full_fn = prefix + "_gauss_poet_results.npz"
        if full_fn in full_fs:
            pairs.append((prefix,
                          os.path.join(data_dir, diag_fn),
                          os.path.join(data_dir, full_fn)))
        else:
            print(f"[WARN] {prefix} 找不到对应的 full 文件：{full_fn}")
    return pairs

def compute_layer_stats(diag_npz, full_npz):
    # 找出所有层号：从 diag 文件中的 layer*_var 键
    var_keys = sorted(
        [k for k in diag_npz.files if k.startswith("layer") and k.endswith("_var")],
        key=lambda k: int(k.split("_")[0].replace("layer", ""))
    )
    mean_rels = []
    max_rels  = []
    layers    = []
    for var_key in var_keys:
        layer = int(var_key.split("_")[0].replace("layer", ""))
        layers.append(layer)
        var_diag = diag_npz[var_key]         # (d+1,)
        cov_key  = f"layer{layer}_Sigma"
        if cov_key not in full_npz.files:
            # 如果 full 中没有 cov，跳过
            mean_rels.append(0.0)
            max_rels.append(0.0)
            continue
        cov      = full_npz[cov_key]         # (d+1, d+1)
        var_full = np.diag(cov)
        nz       = var_diag != 0
        rel      = np.zeros_like(var_diag, dtype=float)
        rel[nz]  = (var_full[nz] - var_diag[nz]) / var_diag[nz] * 100.0  # 百分比
        mean_rels.append(rel[nz].mean() if nz.any() else 0.0)
        max_rels.append(np.max(np.abs(rel[nz])) if nz.any() else 0.0)
    return layers, mean_rels, max_rels

def plot_stats(prefix, layers, mean_rels, max_rels, out_dir):
    plt.figure(figsize=(6,4))
    plt.plot(layers, mean_rels, marker='o', label='Mean rel diff (%)')
    plt.plot(layers, max_rels,  marker='s', label='Max rel diff (%)')
    plt.xlabel("Layer index")
    plt.ylabel("Relative difference (%)")
    plt.title(f"{prefix}")
    plt.xticks(layers)
    plt.legend(loc='best')
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"{prefix}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()

def main():
    pairs = find_result_pairs(DATA_DIR)
    if not pairs:
        print("未找到任何匹配的 diag/full 结果文件，请检查 DATA_DIR。")
        return

    for prefix, diag_path, full_path in pairs:
        print(f"[INFO] 处理实验：{prefix}")
        diag_npz = np.load(diag_path)
        full_npz = np.load(full_path)
        layers, mean_rels, max_rels = compute_layer_stats(diag_npz, full_npz)
        plot_stats(prefix, layers, mean_rels, max_rels, PLOT_DIR)
    print(f"\n所有图片已保存到目录：{os.path.abspath(PLOT_DIR)}")

if __name__ == "__main__":
    main()
