import numpy as np
from sklearn.covariance import LedoitWolf

def run(X_layers, y_all, observed_layers):
    """
    对每层观测的 [w|b] 增广向量做 Ledoit-Wolf 全协方差高斯估计，
    并评估采样向量在全数据集上的分类准确率。

    同时把估计得到的 mu、cov、以及采样得到的 w|b 都保存到 results：
      - layer{l}_acc：shape (M,) 的准确率数组
      - layer{l}_w：  shape (M, d+1) 的采样向量矩阵
      - layer{l}_mu： shape (d+1,) 的估计均值向量
      - layer{l}_cov：shape (d+1, d+1) 的估计协方差矩阵
    """
    n_layers = len(observed_layers)
    results = {}
    sampled_layers = []

    def eval_acc(wb, X, y):
        w = wb[:-1]
        b = wb[-1]
        logits = X.dot(w) + b
        preds = (logits >= 0).astype(int)
        return (preds == y).mean()

    for l in range(n_layers):
        obs = observed_layers[l]             # shape (M, d+1)
        M, dimp1 = obs.shape

        # # 3) 构造空协方差矩阵，并先填对角
        # mu  = obs.mean(axis=0)               # shape (d+1,)
        # var = obs.var(axis=0,ddof=1)

        # cov = np.empty((dimp1, dimp1), dtype=np.float64)
        # for i in range(dimp1):
        #     cov[i, i] = var[i]

        # # 4) 逐对计算协方差（上三角，然后镜像）
        # for i in range(dimp1):
        #     xi = obs[:, i] - mu[i]
        #     for j in range(i+1, dimp1):
        #         xj = obs[:, j] - mu[j]
        #         cov_ij = (xi * xj).sum() / (M - 1)
        #         cov[i, j] = cov[j, i] = cov_ij

        # 1) Ledoit-Wolf 全协方差估计
        lw  = LedoitWolf().fit(obs)
        mu  = lw.location_                   # shape (d+1,)
        cov = lw.covariance_                 # shape (d+1, d+1)

        # 2) 一次性采样 M 条
        samp = np.random.multivariate_normal(mu, cov, size=M)
        sampled_layers.append(samp)

        # 3) 评估准确率
        accs = np.array([eval_acc(samp[i], X_layers[l], y_all) for i in range(M)])

        # —— 保存所有中间量 —— 
        results[f"layer{l}_acc"] = accs
        results[f"layer{l}_w"]   = samp
        results[f"layer{l}_mu"]  = mu
        results[f"layer{l}_cov"] = cov

    return results, sampled_layers

# import numpy as np
# from sklearn.covariance import LedoitWolf

# def run(X_layers, y_all, observed_layers):
#     """
#     对每层观测的 [w|b] 增广向量做 Ledoit-Wolf 全协方差高斯估计，
#     并在估计出的多元高斯分布中、仅在均值 mu 的上下一个标准差范围内进行采样，
#     最后评估采样向量在全数据集上的分类准确率。
    
#     参数：
#       - X_layers: list of np.ndarray，每层的输入特征矩阵，形状为 (N, d)
#       - y_all: np.ndarray，标签向量，长度 N
#       - observed_layers: list of np.ndarray，每层观测到的 [w|b]，形状为 (M, d+1)
    
#     返回:
#       - results: dict, 键为 "layer{l}_acc"，值为长度 M 的准确率数组
#       - sampled_layers: list of np.ndarray，第 l 项为在第 l 层采样得到的 (M, d+1) 矩阵
#     """
#     n_layers = len(observed_layers)
#     results = {}
#     sampled_layers = []

#     # 用于在 X,y 上计算单个 [w|b] 的分类准确率
#     def eval_acc(wb, X, y):
#         w = wb[:-1]     # 前 d 个分量是权重
#         b = wb[-1]      # 最后一个分量是偏置
#         logits = X.dot(w) + b
#         preds = (logits >= 0).astype(int)
#         return (preds == y).mean()

#     for l in range(n_layers):
#         obs = observed_layers[l]           # shape (M, d+1)
#         M, dimp1 = obs.shape              # dimp1 == d+1
#         # 1) 用 Ledoit-Wolf 估计高斯分布的 μ 和 Σ
#         lw = LedoitWolf().fit(obs)
#         mu  = lw.location_                # shape (d+1,)
#         cov = lw.covariance_              # shape (d+1, d+1)
        
#         # 2) 计算每个维度的标准差（协方差矩阵对角线开根号）
#         stds = np.sqrt(np.diag(cov))      # shape (d+1,)

#         # 3) 在 N(mu, cov) 中“超限丢弃”，仅接受每个维度都在 [mu[i]-stds[i], mu[i]+stds[i]] 的样本
#         #    最终采样得到 M 个合法的 [w|b] 向量
#         samp = np.zeros((M, dimp1), dtype=np.float64)
#         count = 0
#         # 当 accepted 个数 < M 时，就持续采样
#         while count < M:
#             # 直接一次性多取一个候选，减少循环开销
#             x = np.random.multivariate_normal(mu, cov)
#             # 检查各维度是否都在 ±1 sigma 之内
#             if np.all(np.abs(x - mu) <= stds):
#                 samp[count] = x
#                 count += 1

#         sampled_layers.append(samp)

#         # 4) 对每个采样向量计算在整层数据上的准确率
#         accs = np.empty(M, dtype=float)
#         for i in range(M):
#             accs[i] = eval_acc(samp[i], X_layers[l], y_all)
#         results[f"layer{l}_acc"] = accs

#     return results, sampled_layers

