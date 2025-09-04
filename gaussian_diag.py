# import numpy as np

# def run(X_layers, y_all, observed_layers):
#     """
#     对每层观测的 [w|b] 增广向量做对角协方差高斯估计，
#     并评估采样向量在全数据集上的分类准确率。
#     同时把估计得到的 mu（均值向量）和 var（方差向量）保存到 results。
#     """
#     n_layers = len(observed_layers)
#     results = {}
#     sampled_layers = []

#     def eval_acc(wb, X, y):
#         w = wb[:-1]
#         b = wb[-1]
#         logits = X.dot(w) + b
#         preds = (logits >= 0).astype(int)
#         return (preds == y).mean()

#     for l in range(n_layers):
#         obs = observed_layers[l]             # shape (M, d+1)
#         M, dimp1 = obs.shape

#         # 1) 估计 mu 和 var（对角协方差假设）
#         mu  = obs.mean(axis=0)               # shape (d+1,)
#         var = obs.var(axis=0,ddof=1)                # shape (d+1,)

#         # 2) 独立高斯采样
#         samp = np.random.randn(M, dimp1) * np.sqrt(var) + mu
#         sampled_layers.append(samp)

#         # 3) 评估准确率
#         accs = np.array([eval_acc(samp[i], X_layers[l], y_all) for i in range(M)])

#         # —— 保存到 results —— 
#         results[f"layer{l}_acc"] = accs       # 原有：准确率
#         results[f"layer{l}_w"]   = samp       # 原有：采样向量
#         results[f"layer{l}_mu"]  = mu         # 新增：估计均值向量
#         results[f"layer{l}_var"] = var        # 新增：估计方差向量

#     return results, sampled_layers


import numpy as np

def run(X_layers, y_all, observed_layers, num=1):
    """
    对每层观测的 [w|b] 增广向量做对角协方差高斯估计环带均匀采样，
    并评估每个采样向量在全数据集上的分类准确率。
    参数:
      - X_layers: list of np.ndarray, 每层特征矩阵，shape (n_samples, d)
      - y_all:     np.ndarray, shape (n_samples,), 全量标签
      - observed_layers: list of np.ndarray, 每层的 [w|b] 向量，shape (M, d+1)
      - num:       环带编号 n，采样范围为 [μ±nσ]，剔除 [μ±(n-1)σ]
    返回:
      - results: dict, 键 “layer{l}_acc”，值 shape=(M,) 的准确率数组
      - sampled_layers: list of np.ndarray, 每层采样得到的 [w|b] 向量，shape (M, d+1)
    """
    n_layers = len(observed_layers)
    results = {}
    sampled_layers = []

    def eval_acc(wb, X, y):
        w = wb[:-1]
        b = wb[-1]
        logits = X.dot(w) + b
        preds  = (logits >= 0).astype(int)
        return (preds == y).mean()

    for l in range(n_layers):
        obs = observed_layers[l]            # shape (M, d+1)
        M, D = obs.shape

        # 1) 计算每维均值和标准差
        mu    = obs.mean(axis=0)           # (D,)
        sigma = obs.std(axis=0)            # (D,)
        var = obs.var(axis=0)
        # 2) 在 ±nσ 超立方体上均匀采样，然后剔除内层 ±(n-1)σ，循环至凑够 M 条
        lower = mu - num * sigma           # (D,)
        upper = mu + num * sigma           # (D,)
        inner_low  = mu - (num-1) * sigma   # (D,)
        inner_high = mu + (num-1) * sigma   # (D,)

        # 用于存放最终合格的 samples
        samp = np.empty((M, D), dtype=float)
        filled = 0

        # 循环直到填满
        while filled < M:
            need = M - filled
            # 在超立方体上均匀采样
            cand = np.random.uniform(low=lower, high=upper, size=(need, D))
            # 筛选出位于第 nσ 环带上的点：至少有一个维度落在外层环带 ([n-1]σ, nσ]
            in_inner = np.all((cand >= inner_low) & (cand <= inner_high), axis=1)
            # valid = not all dims in inner → 至少一个维度超出 inner_high 或低于 inner_low
            valid = ~in_inner
            good  = cand[valid]
            k     = min(len(good), need)
            samp[filled:filled+k] = good[:k]
            filled += k

        sampled_layers.append(samp)

        # 3) 评估这些采样向量在全量数据集上的准确率
        accs = np.array([eval_acc(samp[i], X_layers[l], y_all) for i in range(M)])
        results[f"layer{l}_acc"] = accs       # 原有：准确率
        results[f"layer{l}_w"]   = samp       # 原有：采样向量
        results[f"layer{l}_mu"]  = mu         # 新增：估计均值向量
        results[f"layer{l}_var"] = var        # 新增：估计方差向量
    return results, sampled_layers
