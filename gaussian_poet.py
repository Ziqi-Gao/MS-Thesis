# #!/usr/bin/env python3
# # gauss_poet.py  — POET sampling with rich diagnostics & fine-grained controls
# import numpy as np
# from collections import defaultdict

# # ─────────────────────────────────────────────────────────────
# def run(
#     X_layers, y_all, observed_layers,
#     explained_ratio      = 0.7,     # 若 K_layer_map 不指定，则按该比例自动取 K
#     corr_threshold       = 0.05,    # τ；None→√(log p / M)
#     threshold_mode       = "soft",  # "soft" | "hard"
#     shrink_alpha         = 0.8,     # 全局 shrink；layer_map 可覆写
#     ridge_eps            = 1e-4,    # Σ ← Σ + ε·I 修补正定
#     include_bias_in_poet = True,    # 将 (w,b) 整体估计
#     K_layer_map          = None,    # dict{layer:int -> K} 单独覆盖
#     alpha_layer_map      = None,    # dict{layer:int -> α} 单独覆盖
#     verbose              = True,
# ):
#     """
#     Return
#     -------
#     results : dict
#       layer{i}_acc, layer{i}_w, layer{i}_mu, layer{i}_Sigma
#     sampled_layers : list[np.ndarray]
#     """
#     if K_layer_map is None:     K_layer_map = {}
#     if alpha_layer_map is None: alpha_layer_map = {}

#     results      = {}
#     sampled_list = []
#     n_layers     = len(observed_layers)

#     # ---------- helper ----------
#     def eval_acc(wb, X, y):
#         w, b = wb[:-1], wb[-1]
#         return ((X @ w + b) >= 0).astype(int).mean()

#     # ---------- per-layer loop ----------
#     for l in range(n_layers):
#         obs   = observed_layers[l]          # (M, p=d+1)
#         M, p  = obs.shape
#         Xl    = X_layers[l]
#         d     = p - 1                       # w 维
#         α     = alpha_layer_map.get(l, shrink_alpha)

#         # ---------- 原始统计 ----------
#         w_obs = obs[:, :d]
#         b_obs = obs[:, -1]
#         norm_obs_mean = np.linalg.norm(w_obs, axis=1).mean()
#         #norm_obs_std  = np.linalg.norm(w_obs, axis=1).std()
#         wb_corr = np.corrcoef(w_obs.mean(1), b_obs)[0, 1] if M > 1 else 0.0
#         logit_std_obs = (Xl @ w_obs.T + b_obs).std()

#         # ---------- 准备矩阵 ----------
#         mu = obs.mean(0)
#         obs_c = obs - mu
#         if include_bias_in_poet:
#             Z = obs_c
#         else:
#             Z = obs_c[:, :d]                # 只 w 进 POET
#             b_center = obs_c[:, -1]

#         # ---------- PCA / SVD ----------
#         U, s, Vt = np.linalg.svd(Z, full_matrices=False)
#         S2       = s**2
#         cum_S2   = np.cumsum(S2)
#         K_auto   = np.searchsorted(cum_S2, explained_ratio * cum_S2[-1]) + 1
#         K_sel    = K_layer_map.get(l, K_auto)
#         eigvals  = S2[:K_sel] / (M - 1)
#         eigvecs  = Vt[:K_sel]

#         factor_cov = eigvecs.T * eigvals @ eigvecs

#         # ---------- 残差协方差 ----------
#         sample_cov = np.cov(Z, rowvar=False, ddof=1)
#         Sigma_res  = (sample_cov - factor_cov + sample_cov.T - factor_cov.T) / 2

#         diag_res   = np.maximum(np.diag(Sigma_res), 1e-12)
#         corr_res   = Sigma_res / np.sqrt(diag_res[:, None] * diag_res[None, :])
#         tri_idx    = np.triu_indices(Z.shape[1], k=1)
#         mean_abs_rho = np.abs(corr_res[tri_idx]).mean()
#         frac_small   = np.mean(np.abs(corr_res[tri_idx]) < 0.1)

#         # ---------- 阈值化 ----------
#         if corr_threshold is None:
#             tau = np.sqrt(np.log(p) / M)
#         else:
#             tau = corr_threshold
#         off_diag = Sigma_res - np.diag(np.diag(Sigma_res))
#         if threshold_mode == "soft":
#             over = np.abs(off_diag) - tau
#             off_thr = np.sign(off_diag) * np.maximum(over, 0.0)
#         else:  # hard
#             off_thr = off_diag * (np.abs(off_diag) >= tau)
#         Sigma_res_thr = np.diag(np.diag(Sigma_res)) + off_thr
#         zero_ratio = np.mean((np.abs(off_thr[tri_idx]) < 1e-12))

#         # ---------- Σ_final + 正定修补 ----------
#         Sigma = factor_cov + Sigma_res_thr
#         Sigma = (Sigma + Sigma.T) / 2
#         Sigma += ridge_eps * np.eye(Sigma.shape[1])   # ridge
#         eig, vec = np.linalg.eigh(Sigma)
#         neg_ratio = np.mean(eig <= 0)
#         eig = np.maximum(eig, 1e-8 * eig.max())
#         Sigma = vec * eig @ vec.T
#         L = np.linalg.cholesky(Sigma)

#         # ---------- 采样 ----------
#         M_samp = M
#         Z_g    = np.random.randn(M_samp, Sigma.shape[1])
#         X_samp = α * (Z_g @ L.T) + mu[:Sigma.shape[1]]
#         if include_bias_in_poet:
#             samples = X_samp
#         else:
#             b_samp = np.random.randn(M_samp) * b_center.std(ddof=1) + mu[-1]
#             samples = np.hstack([X_samp, b_samp[:, None]])
#         sampled_list.append(samples)

#         # ---------- diagnostics on samples ----------
#         w_poet = samples[:, :d]
#         norm_poet_mean = np.linalg.norm(w_poet, axis=1).mean()
#         logit_std_poet = (Xl @ w_poet.T + samples[:, -1]).std()

#         # ---------- acc ----------
#         accs = np.array([eval_acc(samples[i], Xl, y_all) for i in range(M_samp)])

#         # ---------- store ----------
#         results[f"layer{l}_acc"]   = accs
#         results[f"layer{l}_w"]     = samples
#         results[f"layer{l}_mu"]    = mu
#         results[f"layer{l}_Sigma"] = Sigma

#         print(f"Layer {l}: K={K_sel} ({cum_S2[K_sel-1]/cum_S2[-1]*100:.1f}% var) | "
#                 f"mean|ρ|={mean_abs_rho:.4f} smallρ%={frac_small*100:.2f} | "
#                 f"zero%={zero_ratio*100:.1f} τ={tau:.3f} | "
#                 f"neg-eig%={neg_ratio*100:.1f}")
#         print(f"         ‖w‖ obs={norm_obs_mean:.3f} poet={norm_poet_mean:.3f} "
#                 f"| logit_std obs={logit_std_obs:.3f} poet={logit_std_poet:.3f} "
#                 f"| corr(mean_w,b)={wb_corr:.3f}")

#     return results, sampled_list








# 只用平均值
# import numpy as np

# def run_mean(X_layers, y_all, observed_layers):
#     """
#     只计算每层观测 [w|b] 的均值向量 μ，用 μ 做预测评估准确率。
#     为兼容 gauss_diag.run 的输出格式：
#       - results["layer{l}_acc"] : shape(M,) 的准确率数组（全是同一个 μ 的 acc）
#       - results["layer{l}_w"]   : shape(M, D)，“采样”矩阵（M 行都为 μ）
#       - results["layer{l}_mu"]  : 均值向量 μ
#       - results["layer{l}_var"] : 原观测向量的方差（可选，但保持键一致）
#     返回:
#       - results: dict
#       - sampled_layers: list[np.ndarray]，同 layer{l}_w
#     """
#     n_layers = len(observed_layers)
#     results = {}
#     sampled_layers = []

#     def eval_acc(wb, X, y):
#         w = wb[:-1]
#         b = wb[-1]
#         logits = X.dot(w) + b
#         preds  = (logits >= 0).astype(int)
#         return (preds == y).mean()

#     for l in range(n_layers):
#         obs = observed_layers[l]          # shape (M, D)
#         M, D = obs.shape

#         mu   = obs.mean(axis=0)           # (D,)
#         var  = obs.var(axis=0)            # (D,) 仅为保持格式

#         # 复制成 M 条，保持接口一致
#         samp = np.repeat(mu[None, :], M, axis=0)  # (M, D)
#         sampled_layers.append(samp)

#         acc_mean = eval_acc(mu, X_layers[l], y_all)
#         accs     = np.full(M, acc_mean, dtype=float)

#         results[f"layer{l}_acc"] = accs
#         results[f"layer{l}_w"]   = samp
#         results[f"layer{l}_mu"]  = mu
#         results[f"layer{l}_var"] = var

#     return results, sampled_layers


# stacking.py
# 功能：对每一层做 two-level stacking：
#       1) 已有内层 1000 个探针的参数 [w_i|b_i]，先把它们拼成 W, b
#       2) 计算所有样本的内层 logits 特征 S = X W^T + b
#       3) 用 StratifiedKFold 产生 out-of-fold 预测，避免信息泄漏
#       4) 在完整 S 上再训练一次外层 LR 得到 (v, b_meta)
#       5) 折回原空间：w_star = W v, b_star = b^T v + b_meta
# 返回：各层最终 [w*|b*] 以及一些指标
#!/usr/bin/env python3
# gauss_stack.py — Stacking-based concept vector estimation
# 兼容 gauss_poet.run 的 I/O：run(X_layers, y_all, observed_layers, ... )
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

def run(
    X_layers, y_all, observed_layers,
    n_splits      = 5,        # K-fold (>=2 开启 out-of-fold 以避免泄漏；=1 时跳过)
    penalty       = "l2",     # 外层 LR 正则
    C             = 1.0,
    max_iter      = 1000,
    solver        = "lbfgs",
    random_state  = 42,
    use_logits    = True,     # True：用 logits S 作为外层特征（保持线性可回溯）；False：用 predict_proba
    verbose       = True,
):
    """
    Parameters
    ----------
    X_layers : list[np.ndarray]
        每层的特征矩阵，shape = (N, d)
    y_all : np.ndarray
        shape = (N,)
    observed_layers : list[np.ndarray]
        每层 M×(d+1) 的观测 [w|b]（内层 1000 × 10% 采样训练得到）
    n_splits : int
        外层 K-fold 的折数。n_splits<=1 时不做 OOF，只全量训练。
    use_logits : bool
        True 时外层使用 logits（推荐；可保持 w_star = W v 线性可回溯）
        False 时会用概率作为外层输入（将失去线性折回的严格性）

    Returns
    -------
    results : dict
        仅包含：
          - layer{i}_acc : float, 用最终 w_star/b_star 在整集上的准确率
          - layer{i}_w   : np.ndarray, shape = (1, d+1)，最终 [w_star|b_star]
    sampled_layers : list[np.ndarray]
        为了与 POET 的 API 对齐，放置每层 shape=(1, d+1) 的 [w_star|b_star]
        （这里并没有“采样”，只是把最终向量以相同的数据结构返回）
    """
    n_layers  = len(observed_layers)
    N         = y_all.shape[0]

    results        = {}
    sampled_layers = []

    # —— 小工具：用 (w*, b*) 评估准确率（与 poet 里的 eval_acc 一致） —— #
    def eval_acc(wb, X, y):
        w, b = wb[:-1], wb[-1]
        return ((X @ w + b) >= 0).astype(int).mean()

    for l in range(n_layers):
        obs  = observed_layers[l]             # (M, d+1)
        M, p = obs.shape
        d    = p - 1

        # 取该层的 X
        Xl = X_layers[l]

        # 拆出所有内层权重矩阵/截距
        W_l = obs[:, :d]                      # (M, d)
        b_l = obs[:, -1]                      # (M,)

        # 计算所有样本在该层的 “基探针 logits 特征” S
        #   S[i,j] = x_i · w_j + b_j
        S = Xl @ W_l.T + b_l                  # (N, M)

        # ===== K-fold 生成 out-of-fold 预测（用于调参 / 验证无泄漏；这里我们只统计一下，不强制返回） =====
        if n_splits is not None and n_splits > 1:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            Z   = np.zeros(N, dtype=np.float32)   # 保存 OOF 的外层 logit（decision_function）
            oof_accs = []
            for tr_idx, vl_idx in skf.split(S, y_all):
                meta_tmp = LogisticRegression(
                    penalty=penalty, C=C, max_iter=max_iter, solver=solver
                )
                # 外层输入
                X_tr = S[tr_idx] if use_logits else _prob(S[tr_idx])
                meta_tmp.fit(X_tr, y_all[tr_idx])

                X_vl = S[vl_idx] if use_logits else _prob(S[vl_idx])
                # decision_function 对 logits 输入是线性项 z
                # 如果你切换到概率特征，这里依然用 decision_function 作为 logit 估计（只是不可再线性折回）
                z_vl = meta_tmp.decision_function(X_vl)
                Z[vl_idx] = z_vl

                # 记录 OOF acc（可选）
                y_pred_vl = (z_vl > 0).astype(int)
                oof_accs.append(accuracy_score(y_all[vl_idx], y_pred_vl))

            if verbose:
                print(f"[Stack][layer {l}] OOF-ACC mean={np.mean(oof_accs):.4f} ± {np.std(oof_accs):.4f}")
        else:
            Z = None

        # ===== 在全量 S 上训练最终外层 LR，得到 v, b_meta =====
        meta_final = LogisticRegression(
            penalty=penalty, C=C, max_iter=max_iter, solver=solver
        )
        X_meta = S if use_logits else _prob(S)
        meta_final.fit(X_meta, y_all)

        v      = meta_final.coef_.ravel()     # (M,)
        b_meta = meta_final.intercept_[0]

        # ===== 折回原始隐藏空间：w_star = W_l^T v,  b_star = b_l^T v + b_meta =====
        if use_logits:
            w_star = W_l.T @ v
            b_star = float(b_l @ v + b_meta)
        else:
            # 如果你真的坚持用概率特征，这里没有线性折回的闭式解；
            # 但你要求 output 只有 w_star，因此建议始终 use_logits=True。
            raise ValueError("use_logits=False 时无法线性折回得到 w_star；请将 use_logits=True")

        wb_star = np.concatenate([w_star, [b_star]])   # shape = (d+1,)

        # ===== 报告最终 acc（用 w_star/b_star 直接在 X_l 上做线性判别） =====
        acc_full = eval_acc(wb_star, Xl, y_all)
        results[f"layer{l}_acc"] = acc_full
        results[f"layer{l}_w"]   = wb_star.reshape(1, -1)

        if verbose:
            print(f"[Stack][layer {l}] acc_full={acc_full:.4f} | "
                  f"M={M}, d={d}, n_splits={n_splits}, penalty={penalty}")

    return results


