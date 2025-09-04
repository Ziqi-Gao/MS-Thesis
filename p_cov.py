#!/usr/bin/env python3
"""
check_cov_anomaly.py
在 lab_rs 目录下扫描 *_gauss_full_results.npz，
检测每层协方差矩阵的异常性。结果打印到终端。
"""
import os, re, glob, numpy as np
import matplotlib.pyplot as plt

LAB_RS_DIR = "./lab_rs"
npz_paths   = glob.glob(os.path.join(LAB_RS_DIR, "*_gauss_full_results.npz"))

def mp_bounds(gamma, sigma2=1.0):
    """Marcenko–Pastur 理论上下界"""
    a = sigma2 * (1 - np.sqrt(gamma))**2
    b = sigma2 * (1 + np.sqrt(gamma))**2
    return a, b

for path in sorted(npz_paths):
    prefix = os.path.basename(path).replace("_gauss_full_results.npz", "")
    data   = np.load(path)
    # 尝试读取同前缀的 diag 文件用来取 var
    diag_path = path.replace("_gauss_full_", "_gauss_diag_")
    diag_data = np.load(diag_path) if os.path.exists(diag_path) else None
    print(f"\n=== {prefix} ===")
    layer_keys = sorted([k for k in data if re.match(r"layer\d+_cov", k)],
                        key=lambda k: int(re.findall(r"layer(\d+)_cov", k)[0]))

    for cov_key in layer_keys:
        l     = int(re.findall(r"layer(\d+)_cov", cov_key)[0])
        cov   = data[cov_key]          # (D, D)
        w_samp = data[f"layer{l}_w"]   # (M, D)
        M, D  = w_samp.shape
        var   = (diag_data[f"layer{l}_var"]
                 if diag_data is not None else np.diag(cov))

        # Z-score 矩阵
        se = np.sqrt( (cov**2 + np.outer(var, var)) / (M - 1) )
        z  = cov / np.maximum(se, 1e-12)
        frac_large = np.mean(np.abs(z[np.triu_indices(D,1)]) > 3)

        # MP 范围
        eigvals = np.linalg.eigvalsh(cov)
        a_mp, b_mp = mp_bounds(gamma=D/M, sigma2=var.mean())
        frac_out = np.mean((eigvals < a_mp) | (eigvals > b_mp))

        # 能量比 & condition number
        offE = np.linalg.norm(cov - np.diag(np.diag(cov)), 'fro')**2
        totE = np.linalg.norm(cov, 'fro')**2
        rho_off = offE / totE
        condnum = eigvals.max() / max(eigvals.min(), 1e-12)

        print(f" layer {l:02d} | "
              f"|Z|>3 比例={frac_large:.3f} | "
              f"MP 超界比={frac_out:.3f} | "
              f"off-energy={rho_off:.3f} | "
              f"cond={condnum:.1e}")

        # 可选：画谱
        # plt.hist(eigvals, bins=50); plt.title(f"{prefix} L{l} spec"); plt.show()
