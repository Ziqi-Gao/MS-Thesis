#!/usr/bin/env python3
# 文件名：main_with_holdout.py
# 功能：在 sampling-train 的基础上，打开 LR 截距，保存 w 和 b，并用它们做 holdout 测试

import argparse
import os
# **在这里设置环境变量，transformers 及 huggingface_hub 都会自动用它来鉴权**
#os.environ["HUGGINGFACE_TOKEN"] = ""
#os.environ["HF_TOKEN"]          = ""

# 然后不要再调用 login() 了，直接让 transformers.from_pretrained 用环境变量：
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

import numpy as np
import torch
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from dataset import DataProcessing
from utils import LLM

def main():
    # —— 参数解析 —— #
    parser = argparse.ArgumentParser()
    parser.add_argument('--savepath',   type=str, default="./lab_rs/", help="结果保存目录")
    parser.add_argument('--model_path', type=str, default="./",      help="模型缓存目录")
    parser.add_argument('--dataset',    type=str, default='STSA',    help="数据集名称")
    parser.add_argument('--datapath',   type=str,
                        default='./dataset/stsa.binary.train',      help="数据文件路径")
    parser.add_argument('--model',      type=str, default='google/gemma-2b',
                        help="LLM 模型标识")
    parser.add_argument('--cuda',       type=int, default=0,         help="CUDA 设备号")
    parser.add_argument('--quant',      type=int, default=32,
                        help="量化位数：8,16,32")
    parser.add_argument('--noise',      type=str, default='non-noise',
                        help="是否添加噪声：noise/non-noise")
    args = parser.parse_args()



    cache_dir = args.model_path
    quant_cfg = None
    if args.quant == 8:
        quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=cache_dir)
    if   args.quant == 32:
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

    # —— 抽取隐藏态 —— #
    DP = DataProcessing(
        data_path=args.datapath,
        data_name=args.dataset,
        noise=args.noise
    )
    pos_q, neg_q, prompt_tmpl, cot = DP.dispacher()
    Model = LLM(cuda_id=args.cuda, layer_num=tot_layer, quant=args.quant)

    # 为每层收集正负样本的最后一个 token 的隐状态
    rp_pos = [[] for _ in range(tot_layer)]
    rp_neg = [[] for _ in range(tot_layer)]
    for samples, storage in [(pos_q, rp_pos), (neg_q, rp_neg)]:
        for q in tqdm(samples, desc="Collecting hidden states"):
            text = DP.get_prompt(prompt_tmpl, cot, q)
            with torch.no_grad():
                hs = Model.get_hidden_states(model, tokenizer, text)  # (layers, seq, dim)
            for l in range(tot_layer):
                vec = hs[l, -1, :].cpu().numpy()
                storage[l].append(vec)

    # 转为 NumPy 矩阵
    X_pos = [np.vstack(rp_pos[l]) for l in range(tot_layer)]
    X_neg = [np.vstack(rp_neg[l]) for l in range(tot_layer)]
    y_pos = np.zeros(len(X_pos[0]), dtype=int)
    y_neg = np.ones(len(X_neg[0]),  dtype=int)
    X_layers = [np.concatenate([X_pos[l], X_neg[l]], axis=0)
                for l in range(tot_layer)]
    y_all     = np.concatenate([y_pos, y_neg], axis=0)

    # —— 采样、训练 & holdout 测试 —— #
    n_iter    = 1000
    n_train   = 692
    dim       = X_layers[0].shape[1]
    W = np.zeros((n_iter, tot_layer, dim), dtype=np.float32)
    B = np.zeros((n_iter, tot_layer),      dtype=np.float32)
    A = np.zeros((n_iter, tot_layer),      dtype=np.float32)

    for it in tqdm(range(n_iter), desc="Sampling & Eval"):
        # 随机选150训练，其余做 holdout
        idx_train = np.random.choice(len(y_all), size=n_train, replace=False)
        mask = np.ones(len(y_all), dtype=bool)
        mask[idx_train] = False
        idx_test = np.nonzero(mask)[0]

        for l in range(tot_layer):
            X_tr, y_tr = X_layers[l][idx_train], y_all[idx_train]
            X_te, y_te = X_layers[l][idx_test],  y_all[idx_test]

            # fit_intercept=True：打开截距
            clf = LogisticRegression(
                penalty='l2'
            )
            clf.fit(X_tr, y_tr)

            # 保存权重向量 w 和截距 b
            W[it, l, :] = clf.coef_.ravel()
            B[it, l]    = clf.intercept_[0]

            # 测试阶段，直接用 w,b 手动预测
            scores = X_te.dot(W[it, l, :]) + B[it, l]
            y_pred = (scores > 0).astype(int)
            A[it, l] = accuracy_score(y_te, y_pred)

    # —— 保存结果 —— #
    os.makedirs(args.savepath, exist_ok=True)
    np.savez(
        os.path.join(args.savepath, 'w_vectors.npz'),
        W=W
    )
    np.savez(
        os.path.join(args.savepath, 'b_intercepts.npz'),
        B=B
    )
    np.savez(
        os.path.join(args.savepath, 'acc_holdout.npz'),
        Acc=A
    )
    print(">>> Saved W, B and Acc to", args.savepath)

if __name__ == "__main__":
    main()














# for i in range(tot_layer):
#     #print("i",i)
#     #rp_log_data_var_name = f'rp_log_data_{i}'
#     #rp_question_data_var_name = f'rp_question_data_{i}'
#     #rp_log_data_i = globals()[rp_log_data_var_name]
#     #rp_question_data_i = globals()[rp_question_data_var_name]
#     rp_log_data_i = rp_log_data_list[i]
#     rp_question_data_i = rp_question_data_list[i]

#     X = np.concatenate((rp_log_data_i, rp_question_data_i), axis=0)
#     y = np.concatenate((labels_log, labels_question), axis=0)
    
#     # Merge data and labels
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

#     # Initialize a random classifier, like Random Forest.
#     if args.clf == 'LR':
#         classifier = LogisticRegression(penalty='l2')
#     elif args.clf == 'RF':
#         classifier = RandomForestClassifier(random_state=42)
#     else:
#         raise ImportError("Cannot use the classifiers that the model haven't imported.")
    
#     classifier.fit(X_train, y_train)
# ###################################################################
#     if args.clf == 'LR':
#         import numpy as np
#         # coef_ 和 intercept_ 在 fit 后才会被创建
#         w = classifier.coef_        # 形状 (1, hidden_size)
#         b = classifier.intercept_   # 形状 (1,)
#         # 直接写你的路径
#         fullpath = "/home/del6500/CD/Concept.npz"
#         np.savez(fullpath, w=w, b=b)
#         print(f"Saved LR params to {fullpath}")
# #######################################################################
#     y_pred = classifier.predict(X_test)
    
#     accuracy = accuracy_score(y_test, y_pred)
#     print("------------------------------------")
#     print("This is epoch",i)
#     print(f'Accuracy: {accuracy}')

#     # Compute and print F1
#     # 'binary' for bi-classification problems; 'micro', 'macro' or 'weighted' for multi-classification problems
#     f1 = f1_score(y_test, y_pred, average='binary')  
#     print(f'F1 Score: {f1}')

#     # Predict the probability of test dataset. (For ROC AUC, we need probabilities instead of label)
#     y_prob = classifier.predict_proba(X_test)[:, 1]  # supposed + class is 1.

#     # Calc and print ROC, AUC
#     roc_auc = roc_auc_score(y_test, y_prob)
#     print(f'ROC AUC Score: {roc_auc}')

#     list_acc.append(accuracy)
#     list_f1.append(f1)
#     list_auc.append(roc_auc)

# # File saving and data
# dict_res = {"Acc":list_acc, "F1":list_f1, "AUC": list_auc}
# def LoadDataset(filename):
#     with open(filename,'r+') as f:
#         read_dict = f.read()
#         f.close()
#     read_dict = json.loads(read_dict)
#     return read_dict

# def SaveDataset(filename, dataset):
#     dict_json = json.dumps(dataset)
#     with open(filename,'w+') as f:
#         f.write(dict_json)
#         f.close()

# model_name_refresh = {"google/gemma-7b":"gemma-7b", "google/gemma-2b": "gemma-2b", "meta-llama/Llama-2-7b-chat-hf": "Llama-7b", "meta-llama/Llama-2-13b-chat-hf":"Llama-13b", "meta-llama/Llama-2-70b-chat-hf":"Llama-70b","Qwen/Qwen1.5-0.5B":"Qwen-0.5B","Qwen/Qwen1.5-1.8B":"Qwen-1.8B","Qwen/Qwen1.5-4B":"Qwen-4B","Qwen/Qwen1.5-7B":"Qwen-7B","Qwen/Qwen1.5-14B":"Qwen-14B","Qwen/Qwen1.5-72B":"Qwen-72B"}
# model_name = model_name_refresh[args.model]

# save_path_final = args.savepath + f"{args.dataset}_{model_name}_{args.quant}_{args.noise}_{args.clf}.json"
# SaveDataset(save_path_final, dict_res)
# print(list_acc)
# print(list_f1)
# print(list_auc)