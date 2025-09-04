#!/usr/bin/env python3
# gauss_stack.py — stacking with internal hold-out test accuracy
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score

def run(
    X_layers, y_all, observed_layers,
    test_frac     = 0.2,     # 在函数内部自动留出 test（最小改动 main 的关键）
    n_splits      = 5,       # 仅在 train 上做 K-fold（避免外层泄漏）
    penalty       = "l2",
    C             = 1.0,
    max_iter      = 1000,
    solver        = "lbfgs",
    random_state  = 42,
    use_logits    = True,    # 必须 True 才能保持 w_star = W v 线性可回溯
    verbose       = True,
):
    """
    Parameters
    ----------
    X_layers : list[np.ndarray]   # 每层 (N, d)
    y_all    : np.ndarray         # (N,)
    observed_layers : list[np.ndarray]  # 每层 (M, d+1) 的 [w|b]

    Returns
    -------
    results : dict
        - layer{i}_w           : (1, d+1)  最终 [w_*|b_*]
        - layer{i}_acc_train   : float     最终向量在内部 train 上的 acc
        - layer{i}_acc_test    : float     最终向量在内部 test 上的 acc
    sampled_layers : list[np.ndarray]
        与 POET 的接口保持一致：每层放一个 (1, d+1) 的 [w_*|b_*]
    """
    if not use_logits:
        raise ValueError("use_logits=False 时无法线性折回得到 w_star；请保持 True。")

    n_layers = len(X_layers)
    results  = {}
    sampled  = []

    # —— 在函数内部做一次全局 hold-out —— #
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=random_state)
    (train_idx, test_idx), = sss.split(np.zeros_like(y_all), y_all)

    y_tr, y_te = y_all[train_idx], y_all[test_idx]
    X_tr_layers = [X[train_idx] for X in X_layers]
    X_te_layers = [X[test_idx]  for X in X_layers]

    for l in range(n_layers):
        # 取该层的数据
        X_tr = X_tr_layers[l]
        X_te = X_te_layers[l]
        obs  = observed_layers[l]              # (M, d+1)

        M, p = obs.shape
        d    = p - 1
        W_l  = obs[:, :d]                      # (M, d)
        b_l  = obs[:, -1]                      # (M,)

        # ———— 先在 train 上构造内层 logits 特征 S_tr ———— #
        S_tr = X_tr @ W_l.T + b_l              # (N_tr, M)

        # ===== K-fold inside TRAIN (避免外层泄漏) =====
        if n_splits > 1 and len(np.unique(y_tr)) > 1:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            oof_accs = []
            for tr_idx, vl_idx in skf.split(S_tr, y_tr):
                meta_tmp = LogisticRegression(
                    penalty=penalty, C=C, max_iter=max_iter, solver=solver
                )
                meta_tmp.fit(S_tr[tr_idx], y_tr[tr_idx])
                z_vl = meta_tmp.decision_function(S_tr[vl_idx])
                y_hat_vl = (z_vl > 0).astype(int)
                oof_accs.append(accuracy_score(y_tr[vl_idx], y_hat_vl))
            if verbose:
                print(f"[Stack][layer {l}] OOF-ACC mean={np.mean(oof_accs):.4f} ± {np.std(oof_accs):.4f}")

        # ===== 在 FULL TRAIN 上训练最终外层，得到 (v, b_meta) =====
        meta_final = LogisticRegression(
            penalty=penalty, C=C, max_iter=max_iter, solver=solver
        )
        meta_final.fit(S_tr, y_tr)
        v      = meta_final.coef_.ravel()      # (M,)
        b_meta = meta_final.intercept_[0]

        # ===== 折回原空间 =====
        w_star = W_l.T @ v                     # (d,)
        b_star = float(b_l @ v + b_meta)
        wb_star = np.concatenate([w_star, [b_star]])  # (d+1,)

        # ===== 统计 train / test acc（用 w_* 直接在原空间判别） =====
        y_hat_tr = (X_tr @ w_star + b_star > 0).astype(int)
        y_hat_te = (X_te @ w_star + b_star > 0).astype(int)
        acc_tr   = accuracy_score(y_tr, y_hat_tr)
        acc_te   = accuracy_score(y_te, y_hat_te)

        results[f"layer{l}_w"]          = wb_star.reshape(1, -1)
        results[f"layer{l}_acc_train"]  = acc_tr
        results[f"layer{l}_acc_test"]   = acc_te

        if verbose:
            print(f"[Stack][layer {l}] acc_train={acc_tr:.4f} | acc_test={acc_te:.4f} "
                  f"| M={M}, d={d}, test_frac={test_frac}")

        sampled.append(wb_star.reshape(1, -1))  # 与 poet 保持同样的 list-of-arrays 返回形式

    return results, sampled
