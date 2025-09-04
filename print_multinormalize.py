# #!/usr/bin/env python3
# # 文件名: multivariate_normality_per_layer.py
# # 功能：对 /home/del6500/CD/Concept.npz 中的 w_all，每一层 (1000×2048) 进行多元正态性检验。
# # 先尝试执行 Henze–Zirkler（若 pingouin >=0.6），再执行 Mardia；
# # 并将每层的检验结果保存到 CSV。

# import numpy as np
# import inspect

# # 尝试导入 pingouin；如未安装请运行 'pip install pingouin'
# try:
#     import pingouin as pg
# except ImportError:
#     raise ImportError("需要安装 pingouin：pip install pingouin 或 conda install -c conda-forge pingouin")

# # 检查 multivariate_normality 签名，判断是否支持 method 参数
# sig = inspect.signature(pg.multivariate_normality)
# supports_method = 'method' in sig.parameters


# def main():
#     # 读取数据
#     arr = np.load('/home/del6500/CD/Gemma2b_STSA_Concept.npz')['w_all']
#     n_samples, n_layers, n_dims = arr.shape
#     print(f"Loaded: {n_samples} samples × {n_layers} layers × {n_dims} dims per layer")

#     results = []
#     for layer in range(n_layers):
#         X = arr[:, layer, :]
#         print(f"\nLayer {layer}:")

#         # Henze–Zirkler (若支持)
#         if supports_method:
#             hz_stat, hz_pval, hz_norm = pg.multivariate_normality(
#                 X, alpha=0.05, method='hz'
#             )
#             print(f"  HZ: stat={hz_stat:.4f}, p={hz_pval:.3e}, normal={hz_norm}")
#         else:
#             hz_stat = hz_pval = np.nan
#             hz_norm = False
#             print("  HZ: skipped (pingouin version <0.6)")

#         # Mardia (默认)
#         m_stat, m_p, m_norm = pg.multivariate_normality(X, alpha=0.05)
#         # 如果返回是 tuple of arrays, unpack
#         try:
#             skew_stat, kurt_stat = m_stat
#             skew_p, kurt_p = m_p
#             skew_norm, kurt_norm = m_norm
#         except Exception:
#             # 旧版可能直接返回 3-tuple for Mardia overall
#             skew_stat = kurt_stat = m_stat
#             skew_p = kurt_p = m_p
#             skew_norm = kurt_norm = m_norm
#         print(f"  Mardia skew: stat={skew_stat:.4f}, p={skew_p:.3e}, normal={skew_norm}")
#         print(f"  Mardia kurt: stat={kurt_stat:.4f}, p={kurt_p:.3e}, normal={kurt_norm}")

#         results.append({
#             'layer': layer,
#             'HZ_stat': hz_stat,
#             'HZ_pval': hz_pval,
#             'HZ_normal': hz_norm,
#             'M_skew_stat': skew_stat,
#             'M_skew_p': skew_p,
#             'M_skew_normal': skew_norm,
#             'M_kurt_stat': kurt_stat,
#             'M_kurt_p': kurt_p,
#             'M_kurt_normal': kurt_norm,
#         })

#     # 保存结果
#     import csv
#     out_csv = '/home/del6500/CD/multivariate_normality_per_layer.csv'
#     fieldnames = [
#         'layer', 'HZ_stat', 'HZ_pval', 'HZ_normal',
#         'M_skew_stat', 'M_skew_p', 'M_skew_normal',
#         'M_kurt_stat', 'M_kurt_p', 'M_kurt_normal'
#     ]
#     with open(out_csv, 'w', newline='') as f:
#         writer = csv.DictWriter(f, fieldnames=fieldnames)
#         writer.writeheader()
#         for r in results:
#             writer.writerow(r)
#     print(f"Results saved to {out_csv}")

# if __name__ == '__main__':
#     main()
import numpy as np
import matplotlib.pyplot as plt

# 准备数据
N = 5000
sigma_val = 2.0

# ① Gaussian ±2σ 截断
gauss = []
while len(gauss) < N:
    pts = np.random.randn(N, 3) * sigma_val
    mask = np.all(np.abs(pts) <= sigma_val, axis=1)
    gauss.append(pts[mask])
gauss = np.vstack(gauss)[:N]

# ② Uniform 壳层
uniform = np.random.uniform(-sigma_val, sigma_val, (N, 3))

# 只看 X–Y 平面
gxy = gauss[:, :2]
uxy = uniform[:, :2]

# 画 hexbin 
fig, axes = plt.subplots(1, 2, figsize=(10,4), sharex=True, sharey=True)

# Gaussian
hb1 = axes[0].hexbin(gxy[:,0], gxy[:,1], gridsize=50, cmap='Blues')
axes[0].set_title("Gaussian ±2σ\nHexbin density")
axes[0].set_aspect('equal')
fig.colorbar(hb1, ax=axes[0])

# Uniform
hb2 = axes[1].hexbin(uxy[:,0], uxy[:,1], gridsize=50, cmap='Oranges')
axes[1].set_title("Uniform Cube Shell ±2σ\nHexbin density")
axes[1].set_aspect('equal')
fig.colorbar(hb2, ax=axes[1])

plt.tight_layout()
plt.savefig("density_compare_hexbin.png", dpi=300)
plt.show()



