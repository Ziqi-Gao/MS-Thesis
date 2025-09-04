# #!/usr/bin/env python3
# """
# analyze_correlations.py

# 功能：
# 1. 从给定的 .npz 文件中加载权重张量 (shape: n_iter × num_layers × dim)
# 2. 对每一层计算：
#    a) 两两 Pearson 相关系数矩阵 R
#    b) 全层平均绝对相关度 mean_abs_layer[l]
#    c) 每个维度的平均绝对相关度 mean_abs_per_dim[l, :]
# 3. 可视化：
#    - 总览：mean_abs_layer 折线图
#    - 每层：
#      1) 维度平均相关度直方图
#      2) 相关矩阵热图

# 用法示例：
#     python p_corr_plot.py \
#       --data /gpfs/projects/p32737/del6500_home/CD/lab_rs/gemma2b_cities_692_w.npz \
#       --output_dir ./corr_result
# """
# import os
# import argparse
# import numpy as np
# import matplotlib.pyplot as plt

# def main():
#     parser = argparse.ArgumentParser(
#         description='Analyse correlations in concept weight vectors per layer.'
#     )
#     parser.add_argument('--data', type=str, required=True,
#                         help='Path to .npz containing w_all or W array')
#     parser.add_argument('--output_dir', type=str, default='./corr_result',
#                         help='Directory to save figures')
#     args = parser.parse_args()

#     os.makedirs(args.output_dir, exist_ok=True)

#     # 加载 W
#     data = np.load(args.data)
#     if 'w_all' in data:
#         W = data['w_all']
#     elif 'W' in data:
#         W = data['W']
#     else:
#         raise KeyError(f"数据键必须包含 'w_all' 或 'W'，当前 keys: {data.files}")

#     n_iter, num_layers, dim = W.shape
#     print(f"Loaded W shape: {W.shape}")

#     # 初始化容器
#     mean_abs_layer = np.zeros(num_layers)
#     mean_abs_per_dim = np.zeros((num_layers, dim))

#     # 逐层计算相关度，保存统计
#     print("Computing correlations for each layer...")
#     for l in range(num_layers):
#         Wl = W[:, l, :]
#         R = np.corrcoef(Wl, rowvar=False)
#         R = np.nan_to_num(R)
#         np.fill_diagonal(R, 0)
#         mean_abs_layer[l] = np.mean(np.abs(R))
#         mean_abs_per_dim[l] = np.mean(np.abs(R), axis=1)

#     # 1. 总览：可视化每层平均绝对相关度
#     fig1 = os.path.join(args.output_dir, 'mean_abs_corr_per_layer.png')
#     plt.figure()
#     plt.plot(range(num_layers), mean_abs_layer, marker='o')
#     plt.xlabel('Layer Index')
#     plt.ylabel('Mean Absolute Correlation')
#     plt.title('Mean Abs Correlation per Layer')
#     plt.grid()
#     plt.savefig(fig1)
#     plt.close()
#     print(f"Saved overview plot: {fig1}")

#     # 计算置信区间阈值
#     N = n_iter
#     thresh95 = 1.96 / np.sqrt(N - 2)
#     thresh99 = 2.58 / np.sqrt(N - 2)

#     # 2. 为每一层生成直方图和热图
#     for l in range(num_layers):
#         print(f"Processing visualization for layer {l}...")
#         # 2.1 直方图
#         vals = mean_abs_per_dim[l]
#         vals = vals[np.isfinite(vals)]
#         plt.figure()
#         plt.hist(vals, bins=50, edgecolor='k', alpha=0.7)
#         plt.axvline(thresh95, color='red', linestyle='--',
#                     label=f'95% CI |ρ| < {thresh95:.3f}')
#         plt.axvline(thresh99, color='purple', linestyle=':',
#                     label=f'99% CI |ρ| < {thresh99:.3f}')
#         plt.xlabel('Mean Abs Corr')
#         plt.ylabel('Count')
#         plt.title(f'Layer {l}: Histogram of Mean Abs Corr per Dim')
#         plt.legend()
#         hist_path = os.path.join(args.output_dir,
#                                  f'layer_{l}_hist_mean_abs_corr.png')
#         plt.savefig(hist_path)
#         plt.close()
#         print(f"Saved: {hist_path}")

#         # 2.2 热图
#         Wl = W[:, l, :]
#         R = np.corrcoef(Wl, rowvar=False)
#         R = np.nan_to_num(R)
#         np.fill_diagonal(R, 0)
#         plt.figure(figsize=(6, 6))
#         plt.imshow(R, cmap='coolwarm', aspect='auto', interpolation='none')
#         plt.colorbar()
#         plt.xlabel('Dim Index')
#         plt.ylabel('Dim Index')
#         plt.title(f'Layer {l}: Correlation Matrix Heatmap')
#         heatmap_path = os.path.join(args.output_dir,
#                                     f'layer_{l}_corr_heatmap.png')
#         plt.savefig(heatmap_path)
#         plt.close()
#         print(f"Saved: {heatmap_path}")

#     # 输出目录内容
#     print('Output directory listing:')
#     for root, dirs, files in os.walk(args.output_dir):
#         print(f"{root}: {len(files)} files -> {files}")

# if __name__ == '__main__':
#     main()










import numpy as np
import matplotlib.pyplot as plt
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from scipy.stats import chi2

# --- Step 1: Load data and select a layer ---
filename = "/gpfs/projects/p32737/del6500_home/CD/lab_rs/google_gemma-2b_cities_149_w.npz"  # TODO: replace with your NPZ file path
layer_index = 9  # default to second-last layer (user can change this index as needed)

# Load the NPZ file. If the NPZ contains a single array, we'll extract it.
data = np.load(filename)
# If multiple arrays are present, adjust accordingly. Here we assume one main array.
arr_keys = list(data.keys())
print(f"Keys in NPZ file: {arr_keys}")  # Debug: list keys to understand the structure
X_all_layers = data[arr_keys[0]]  # assume the first array contains the data of shape (N, n_layers, dim)

# Verify shape and select the specified layer
N, n_layers, dim = X_all_layers.shape
print(f"Loaded data shape: {X_all_layers.shape} (N, n_layers, dim)")
print(f"Selecting layer index {layer_index} (layer {layer_index} out of {n_layers-1} indexed from 0)") 
X_layer = X_all_layers[:, layer_index, :]  # shape (N, dim), data from the chosen layer

# (Optional) If needed, allow custom layer selection:
# layer_index = int(input("Enter layer index to analyze: "))  # in interactive use
# X_layer = X_all_layers[:, layer_index, :]


# --- Step 2: Standardize data and compute shrunk covariance & correlation matrix ---
X = X_layer  # rename for convenience
# Remove any features with zero variance to avoid division by zero
stds = X.std(axis=0, ddof=0)
zero_var_mask = (stds == 0)
if np.any(zero_var_mask):
    print(f"Warning: {np.sum(zero_var_mask)} dimensions have zero variance and will be excluded from analysis.")
    X = X[:, ~zero_var_mask]
    stds = stds[~zero_var_mask]
    dim = X.shape[1]  # update dimension count after dropping constant features

# Standardize features to mean=0, variance=1 (so covariance = correlation)
means = X.mean(axis=0)
X_std = (X - means) / stds

# Use Ledoit-Wolf shrinkage to estimate covariance matrix (to handle p >> N).
lw = LedoitWolf().fit(X_std)  # fit on standardized data
cov_matrix = lw.covariance_   # shrunk covariance matrix
shrinkage_alpha = lw.shrinkage_  # shrinkage coefficient used
print(f"Ledoit-Wolf shrinkage alpha: {shrinkage_alpha:.4f}")

# Convert covariance to correlation matrix (should be close to correlation since X_std is standardized)
# We ensure diagonal is 1 and compute off-diagonals as cov/sigma_i/sigma_j.
diag = np.sqrt(np.diag(cov_matrix))
corr_matrix = cov_matrix / np.outer(diag, diag)
corr_matrix[np.diag_indices_from(corr_matrix)] = 1.0  # set diagonal exactly to 1

# --- Step 3: Bartlett's test for sphericity (independence of dimensions) ---
# Hypothesis H0: correlation matrix is identity (features are independent).
# Compute test statistic and p-value.
N_samples = X_std.shape[0]
p = X_std.shape[1]
# Compute log determinant of correlation matrix in a stable way
sign, logdet = np.linalg.slogdet(corr_matrix)
if sign <= 0:
    print("Correlation matrix is singular or not positive-definite; Bartlett's test cannot be applied.")
else:
    # Bartlett's test statistic (chi-square distributed with df = p*(p-1)/2 under H0)
    statistic = - (N_samples - 1 - (2*p + 5) / 6) * logdet
    dof = p * (p - 1) / 2  # degrees of freedom
    p_value = 1 - chi2.cdf(statistic, dof)
    print(f"Bartlett's test statistic: {statistic:.4f}, degrees of freedom: {dof:.1f}")
    if p_value < 1e-16:
        print(f"Bartlett's test p-value: <1e-16 (very small, effectively 0)")
    else:
        print(f"Bartlett's test p-value: {p_value:.4f}")
    # Interpretation: A very small p-value (e.g., < 0.05) means we reject the null hypothesis
    # that the correlation matrix is identity. This indicates the dimensions are not independent.


# --- Step 4: Visualize correlation matrix as heatmap ---
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving figures (especially on HPC systems)
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 6))
cax = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
ax.set_title(f'Correlation Matrix (layer {layer_index})')
# Remove tick labels for clarity in a large matrix
ax.set_xticks([])
ax.set_yticks([])
fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig("correlation_matrix_heatmap.png", dpi=300)
plt.close(fig)
print("Correlation matrix heatmap saved as 'correlation_matrix_heatmap.png'.")


# --- Step 5: Find and output top-K correlated dimension pairs ---
K = 20  # number of top pairs to display
corr_matrix_offdiag = corr_matrix.copy()
np.fill_diagonal(corr_matrix_offdiag, 0.0)  # ignore self-correlation on the diagonal
# Get absolute correlations and their indices for the upper triangle (to avoid duplicate pairs)
p = corr_matrix_offdiag.shape[0]
upper_indices = np.triu_indices(p, k=1)
abs_corr_vals = np.abs(corr_matrix_offdiag[upper_indices])
# Find indices of the top K correlations
if K > len(abs_corr_vals):
    K = len(abs_corr_vals)
topK_idx = np.argpartition(abs_corr_vals, -K)[-K:]  # indices in the flattened upper-tri array
topK_sorted_idx = topK_idx[np.argsort(abs_corr_vals[topK_idx])[::-1]]  # sort these indices by correlation
top_pairs = []  # will store (corr_value, dim_i, dim_j)
for idx in topK_sorted_idx:
    i = upper_indices[0][idx]
    j = upper_indices[1][idx]
    corr_val = corr_matrix_offdiag[i, j]
    top_pairs.append((corr_val, i, j))

print(f"Top {K} most strongly correlated dimension pairs (using 0-based dimension indices):")
for corr_val, i, j in top_pairs:
    print(f" Dimension {i} & Dimension {j}: correlation = {corr_val:.3f}")


# --- Step 6: PCA to analyze main variance directions ---
pca = PCA()  # by default, n_components = min(N_samples, n_features) will be used
pca.fit(X_std)
explained_variances = pca.explained_variance_       # variance of each PC
explained_variance_ratio = pca.explained_variance_ratio_  # fraction of total variance
num_components = explained_variance_ratio.shape[0]

# Print how many components are needed for certain variance thresholds
cumulative_variance = np.cumsum(explained_variance_ratio)
for threshold in [0.8, 0.9, 0.95]:
    comp_count = np.searchsorted(cumulative_variance, threshold) + 1
    if comp_count <= num_components:
        print(f"{threshold*100:.0f}% of variance is explained by the first {comp_count} principal components.")
    else:
        print(f"{threshold*100:.0f}% of variance is not reached even with all {num_components} components.")

# Plot explained variance ratio for the first 50 components (or all, if fewer than 50)
n_plot = min(50, num_components)
fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.plot(range(1, n_plot+1), explained_variance_ratio[:n_plot]*100, marker='o')
ax2.set_xlabel('Principal Component')
ax2.set_ylabel('Variance Explained (%)')
ax2.set_title('Explained Variance by Principal Components (first {} PCs)'.format(n_plot))
ax2.grid(True)
plt.tight_layout()
plt.savefig("pca_explained_variance.png", dpi=300)
plt.close(fig2)
print(f"PCA analysis done. Explained variance plot saved as 'pca_explained_variance.png'.")
