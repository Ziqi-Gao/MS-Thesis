#!/usr/bin/env python3
# 文件名：main.py
import argparse
import os
import numpy as np
import torch
import logging
logger = logging.getLogger(__name__)
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from dataset import DataProcessing
from util import LLM
from sklearn.model_selection import train_test_split
# 导入 Gaussian 模块
import gaussian_diag
import gaussian_full
import gaussian_poet
import gaussian_stacking
import warnings
from sklearn.exceptions import ConvergenceWarning
import pickle,pathlib
# 在程序最开始，越早越好
warnings.filterwarnings("ignore", category=ConvergenceWarning)
def main():
    # —— 参数解析 —— #
    parser = argparse.ArgumentParser()
    parser.add_argument('--savepath',   type=str, default="./lab_rs/", help="结果保存目录")
    parser.add_argument('--model_path', type=str, default="./",      help="模型缓存目录")
    parser.add_argument('--dataset',    type=str, default='STSA',    help="数据集名称")
    parser.add_argument('--datapath',   type=str,
                        default='./dataset/stsa.binary.train', help="数据文件路径")
    parser.add_argument('--model',      type=str, default='google/gemma-2b',
                        help="LLM 模型标识")
    parser.add_argument('--cuda',       type=int, default=0,         help="CUDA 设备号")
    parser.add_argument('--quant',      type=int, default=32,
                        help="量化位数：8,16,32")
    parser.add_argument('--noise',      type=str, default='non-noise',
                        help="是否添加噪声：noise/non-noise")
    parser.add_argument('--use_diag',   action='store_true', help="是否运行对角协方差Gaussian分析")
    parser.add_argument('--use_full',   action='store_true', help="是否运行全协方差Gaussian分析")
    parser.add_argument('--use_poet',   action='store_true', help="是否运行POET分析")
    parser.add_argument('--use_stacking',   action='store_true', help="是否运行stacking分析")
    parser.add_argument('--concept',type=str,default='',help='要处理的概念名称（例如 Bird），脚本会自动去 dataset/raw/{concept}.pkl')
    
    args = parser.parse_args()

    # —— 准备模型 —— #
    #os.environ["HUGGINGFACE_TOKEN"] = os.environ["HF_TOKEN"] = "yourtoken"
    cache_dir = args.model_path
    quant_cfg = None
    if args.quant == 8:
        quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=cache_dir)
    if args.quant == 32:
        model = AutoModelForCausalLM.from_pretrained(args.model, cache_dir=cache_dir)
    elif args.quant == 8:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=quant_cfg, cache_dir=cache_dir
        )
    else:  # 16-bit
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, cache_dir=cache_dir
        )

    # —— 层数映射 —— #
    layer_map = {
        "google/gemma-2b":18, "google/gemma-7b":28,
        "meta-llama/Llama-2-7b-chat-hf":32, "meta-llama/Llama-2-13b-chat-hf":40,
        "meta-llama/Llama-2-70b-chat-hf":80,
        "Qwen/Qwen1.5-0.5B":24, "Qwen/Qwen1.5-1.8B":24,
        "Qwen/Qwen1.5-4B":40,   "Qwen/Qwen1.5-7B":32,
        "Qwen/Qwen1.5-14B":40,  "Qwen/Qwen1.5-72B":80
    }
    tot_layer = layer_map[args.model]
        # ——【改动开始】—————————————————————————————————————
        # ——【数据读取改造】—————————————————————————————————————
    if args.concept:
        # 根据概念名自动拼出路径
        one_pkl = os.path.join('dataset', 'raw', f"{args.concept}.pkl")
        if not os.path.isfile(one_pkl):
            raise FileNotFoundError(f"{one_pkl} 不存在，请检查概念名称")
        with open(one_pkl, 'rb') as f:
            d = pickle.load(f)
        key = list(d.keys())[0]
        pos_q = d[key].get('positive', [])
        neg_q = d[key].get('negative', [])
        prompt_tmpl = "{text}"
        cot = False
        dataset_tag = args.concept
        print(f"✅ 加载 {args.concept}: 正例 {len(pos_q)} 条, 负例 {len(neg_q)} 条")
        print(f"[{args.concept}] 正样本 {len(pos_q)} 条 | 负样本 {len(neg_q)} 条")
    else:                            # 默认走 DataProcessing
        DP = DataProcessing(
            data_path=args.datapath,
            data_name=args.dataset,
            noise=args.noise
        )
        pos_q, neg_q, prompt_tmpl, cot = DP.dispacher()
    Model = LLM(cuda_id=args.cuda, layer_num=tot_layer, quant=args.quant)

    rp_pos = [[] for _ in range(tot_layer)]
    rp_neg = [[] for _ in range(tot_layer)]
    for samples, storage in [(pos_q, rp_pos), (neg_q, rp_neg)]:
        for q in tqdm(samples, desc="Collecting hidden states"):
            text = q #if args.concept else DP.get_prompt(prompt_tmpl, cot, q)
            with torch.no_grad():
                hs = Model.get_hidden_states(model, tokenizer, text)  # (layers, seq, dim)
            for l in range(tot_layer):
                vec = hs[l, -1, :].cpu().numpy()
                storage[l].append(vec)

    # # 转为 NumPy 矩阵
    # X_pos = [np.vstack(rp_pos[l]) for l in range(tot_layer)]
    # X_neg = [np.vstack(rp_neg[l]) for l in range(tot_layer)]
    # y_pos = np.zeros(len(X_pos[0]), dtype=int)
    # y_neg = np.ones(len(X_neg[0]),  dtype=int)
    # X_layers = [np.concatenate([X_pos[l], X_neg[l]], axis=0) for l in range(tot_layer)]
    # y_all     = np.concatenate([y_pos, y_neg], axis=0)
    # —— 转为 NumPy 矩阵 —— #
    X_pos = [np.vstack(rp_pos[l]) for l in range(tot_layer)]
    X_neg = [np.vstack(rp_neg[l]) for l in range(tot_layer)]
    y_pos = np.ones(len(X_pos[0]),  dtype=int)   # 正例用 1
    y_neg = np.zeros(len(X_neg[0]), dtype=int)   # 负例用 0
    X_layers = [np.concatenate([X_pos[l], X_neg[l]], axis=0) for l in range(tot_layer)]
    y_all     = np.concatenate([y_pos, y_neg], axis=0)

    # # —— 训练 & holdout 测试 —— #
    # n_total = len(y_all)
    # # 1/5 的正例数、1/5 的负例数
    # n_pos   = len(y_pos) // 5
    # n_neg   = len(y_neg) // 5
    # n_iter  = 1000
    # dim     = X_layers[0].shape[1]
    # max_iter = 1000 

    # —— 训练 & holdout 测试 —— #
    n_total = len(y_all)
    # 不区分正/负例，训练集大小为总样本数的 10%
    n_train = max(1, n_total // 10)
    n_iter  = 1000
    dim     = X_layers[0].shape[1]
    max_iter = 1000 




    W = np.zeros((n_iter, tot_layer, dim), dtype=np.float32)
    B = np.zeros((n_iter, tot_layer),      dtype=np.float32)
    A = np.zeros((n_iter, tot_layer),      dtype=np.float32)
    # 超参，完全按照作者设置
    MAX_ITER   = 100
    EARLY_LOOPS = 10   # 最多重试次数
    VAL_THRESH = 0.90

    # for it in tqdm(range(n_iter), desc="Sampling & Eval"):
    #     # —— 抽样训练子集（1/5 positive, 1/5 negative），保留 idx_test 用于最终测试 —— #
    #     pos_idx = np.arange(len(X_pos[0]))
    #     neg_idx = np.arange(len(X_pos[0]), len(X_pos[0]) + len(X_neg[0]))
    #     sel_pos = np.random.choice(pos_idx, size=n_pos, replace=True)
    #     sel_neg = np.random.choice(neg_idx, size=n_neg, replace=True)
    #     idx_train = np.concatenate([sel_pos, sel_neg])
    #     mask = np.ones(n_total, dtype=bool); mask[idx_train] = False
    #     idx_test  = np.nonzero(mask)[0]



    #GCS training algorithm
    for it in tqdm(range(n_iter), desc="Sampling & Eval"):
        # —— 不区分类别，从所有样本中随机抽取 10% 作为训练集 —— #
        idx_train = np.random.choice(n_total, size=n_train, replace=True)
        mask = np.ones(n_total, dtype=bool)
        mask[idx_train] = False
        idx_test = np.nonzero(mask)[0]
        for l in range(tot_layer):
            # 原始子集
            X_sub = X_layers[l][idx_train];  y_sub = y_all[idx_train]
            X_test = X_layers[l][idx_test]; y_test = y_all[idx_test]
        


            # 在子集上再划分 train/val/test
            X_tr, X_tmp, y_tr, y_tmp = train_test_split(
                X_sub, y_sub, test_size=0.3, random_state=None, shuffle=True
            )
            # tmp 里 1/3 做真正的 test，2/3 做 val
            X_val, X_valtest, y_val, y_valtest = train_test_split(
                X_tmp, y_tmp, test_size=0.66, random_state=None, shuffle=True
            )

            # 标准化
            scaler = StandardScaler().fit(X_tr)
            X_tr   = scaler.transform(X_tr)
            X_val  = scaler.transform(X_val)
            X_test = scaler.transform(X_test)

            best_coef     = None
            best_intercept = None
            best_acc_test = 0.0
            converged     = False

            # 早停循环
            for _ in range(EARLY_LOOPS):
                clf = LogisticRegression(
                    penalty='l2',
                    solver='lbfgs',
                    fit_intercept=True,
                    max_iter=MAX_ITER
                )
                clf.fit(X_tr, y_tr)

                # 是否“真正”收敛
                nit = clf.n_iter_
                if isinstance(nit, (list, np.ndarray)):
                    nit = max(nit)
                did_conv = (nit < MAX_ITER)

                # 评估 val & test
                val_acc  = accuracy_score(y_val,  clf.predict(X_val))
                test_acc = accuracy_score(y_test, clf.predict(X_test))

                # 达到收敛或 Val 阈值则 early stop
                if did_conv or val_acc > VAL_THRESH:
                    best_coef     = clf.coef_.ravel().copy()
                    best_acc_test = test_acc
                    best_intercept = clf.intercept_[0]
                    converged     = did_conv
                    break

                # 否则保留当前最优
                best_coef     = clf.coef_.ravel().copy()
                best_acc_test = test_acc
                best_intercept = clf.intercept_[0]
            # 如果完全没收敛且 val 低于阈值，打个警告
            if not converged and val_acc <= VAL_THRESH:
                logger.warning(
                    f"Layer {l}, split {it}: neither converged in {MAX_ITER} iters nor "
                    f"val_acc>{VAL_THRESH:.2f} (val_acc={val_acc:.3f}, n_iter={nit})"
                )
            else:
                logger.warning(
                    f" Converged Successful "
                )
            # 保存 w, b, 并在 hold‑out 测试集上记录准确率
            W[it, l, :] = best_coef
            B[it, l]    = best_intercept  
            A[it, l]    = best_acc_test


    # # —— 训练 & holdout 测试 —— #
    # n_total = len(y_all)
    # n_train = max(1, n_total // 10)
    # n_iter  = 1000
    # dim     = X_layers[0].shape[1]
    # W = np.zeros((n_iter, tot_layer, dim), dtype=np.float32)
    # B = np.zeros((n_iter, tot_layer),      dtype=np.float32)
    # A = np.zeros((n_iter, tot_layer),      dtype=np.float32)

    # for it in tqdm(range(n_iter), desc="Sampling & Eval"):
    #     idx_train = np.random.choice(n_total, size=n_train, replace=True)
    #     mask = np.ones(n_total, dtype=bool)
    #     mask[idx_train] = False
    #     idx_test = np.nonzero(mask)[0]
    #     for l in range(tot_layer):
    #         X_tr, y_tr = X_layers[l][idx_train], y_all[idx_train]
    #         X_te, y_te = X_layers[l][idx_test],  y_all[idx_test]
    #         clf = LogisticRegression(penalty='l2', max_iter=1000)
    #         clf.fit(X_tr, y_tr)
    #         # W[it, l, :] = clf.coef_.ravel()
    #         # B[it, l]    = clf.intercept_[0]
    #         # scores = X_te.dot(W[it, l, :]) + B[it, l]
    #         # A[it, l]    = accuracy_score(y_te, (scores > 0).astype(int))
    #         w = clf.coef_.ravel()
    #         b = clf.intercept_[0]
    #                 # 保存参数
    #         W[it, l, :] = w
    #         B[it, l]    = b

    #         # 训练/测试准确率
    #         pred_tr = (X_tr.dot(w) + b > 0).astype(int)
    #         pred_te = (X_te.dot(w) + b > 0).astype(int)
    #         acc_tr  = accuracy_score(y_tr, pred_tr)
    #         acc_te  = accuracy_score(y_te, pred_te)
    #         A[it, l] = acc_te

    #         # 收敛判断
    #         nit = clf.n_iter_
    #         if isinstance(nit, (list, np.ndarray)):
    #             nit = max(nit)
    #         did_conv = (nit < MAX_ITER)

    #         # 日志输出
    #         if did_conv:
    #             logger.info(f"[it {it:04d} | layer {l:02d}] train_acc={acc_tr:.3f}, test_acc={acc_te:.3f}, n_iter={nit} (Converged)")
    #         else:
    #             logger.warning(
    #                 f"[it {it:04d} | layer {l:02d}] train_acc={acc_tr:.3f}, test_acc={acc_te:.3f}, n_iter={nit} "
    #                 f"(NOT converged in {MAX_ITER} iters)"
    #             )

    # —— 保存 w, b, acc —— #
    os.makedirs(args.savepath, exist_ok=True)
    model_tag   = args.model.replace('/', '-')
    dataset_tag = args.concept if args.concept else args.dataset
    base_name   = f"{model_tag}_{dataset_tag}"
    np.savez(os.path.join(args.savepath, f"{base_name}_w.npz"), W=W)
    np.savez(os.path.join(args.savepath, f"{base_name}_b.npz"), B=B)
    np.savez(os.path.join(args.savepath, f"{base_name}_acc.npz"), Acc=A)
    print(f">>> Saved: {base_name}_w.npz, {base_name}_b.npz, {base_name}_acc.npz")

    # 构造每层的观测增广向量 [w|b]
    observed_layers = []
    for l in range(tot_layer):
        w_mat = W[:, l, :]            # shape (1000, d)
        b_vec = B[:, l].reshape(-1,1) # shape (1000, 1)
        observed_layers.append(np.hstack([w_mat, b_vec]))  # (1000, d+1)

    # 如需 Gaussian 分析，创建子目录
    if args.use_diag or args.use_full:
        os.makedirs(os.path.join(args.savepath, "obs"), exist_ok=True)
        os.makedirs(os.path.join(args.savepath, "samp"), exist_ok=True)

    # 对角协方差 Gaussian
    # gauss_diag 部分
    if args.use_diag:
        results_diag, sampled_diag = gaussian_diag.run(X_layers, y_all, observed_layers)
        # 把所有 layer*_acc, layer*_w, layer*_mu, layer*_var 一起打包
        np.savez(
            os.path.join(args.savepath, f"{base_name}_gauss_diag_results.npz"),
            **results_diag
        )

        # 保存观测 & 采样向量
        for l in range(tot_layer):
            np.save(os.path.join(args.savepath, "obs",  f"{base_name}_layer{l}_diag_obs.npy"),
                    observed_layers[l])
            np.save(os.path.join(args.savepath, "samp", f"{base_name}_layer{l}_diag_samp.npy"),
                    sampled_diag[l])

    # Ledoit-Wolf 全协方差 Gaussian
    # gauss_full 部分
    if args.use_full:
        results_full, sampled_full = gaussian_full.run(X_layers, y_all, observed_layers)
        # 把所有 layer*_acc, layer*_w, layer*_mu, layer*_cov 一起打包
        np.savez(
            os.path.join(args.savepath, f"{base_name}_gauss_full_results.npz"),
            **results_full
        )
        for l in range(tot_layer):
            np.save(os.path.join(args.savepath, "obs",  f"{base_name}_layer{l}_full_obs.npy"),
                    observed_layers[l])
            np.save(os.path.join(args.savepath, "samp", f"{base_name}_layer{l}_full_samp.npy"),
                    sampled_full[l])
    # POET
    # gauss_poet 部分
    if args.use_poet:
        results_poet, sampled_poet = gaussian_poet.run(X_layers, y_all, observed_layers)
        # 把所有 layer*_acc, layer*_w, layer*_mu, layer*_var 一起打包
        np.savez(
            os.path.join(args.savepath, f"{base_name}_gauss_poet_results.npz"),
            **results_poet
        )

        # 保存观测 & 采样向量
        for l in range(tot_layer):
            np.save(os.path.join(args.savepath, "obs",  f"{base_name}_layer{l}_poet_obs.npy"),
                    observed_layers[l])
            np.save(os.path.join(args.savepath, "samp", f"{base_name}_layer{l}_poet_samp.npy"),
                    sampled_poet[l])
    if args.use_stacking:
        results_stacking, sampled_stacking = gaussian_stacking.run(X_layers, y_all, observed_layers)
        # 把所有 layer*_acc, layer*_w, layer*_mu, layer*_var 一起打包
        np.savez(
            os.path.join(args.savepath, f"{base_name}_gauss_stacking_results.npz"),
            **results_stacking
        )
       
        #保存观测 & 采样向量
        for l in range(tot_layer):
            np.save(os.path.join(args.savepath, "samp", f"{base_name}_layer{l}_stacking_samp.npy"),
                    sampled_stacking[l])
    

if __name__ == "__main__":
    main()
