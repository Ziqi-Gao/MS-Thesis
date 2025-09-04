# # -*- coding: utf-8 -*-
# """
# 功能：按论文式(11)进行 activation steering，只负责“生成与落盘”，不做 GPT 评估
# 输出：CSV，列为 [method, alpha, prompt, original_output, steered_output]
# """

# import os
# import re
# import csv
# import argparse
# import numpy as np
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM

# # -----------------------
# # 工具：按层号排序文件名
# # -----------------------
# def sort_by_layer(files):
#     return sorted(files, key=lambda x: int(re.findall(r'\d+', x)[-1]) if re.findall(r'\d+', x) else x)

# def to_model_tag(model_or_dir: str) -> str:
#     """
#     将模型名/本地目录转成文件前缀，如:
#       "google/gemma-2b"           -> "google-gemma-2b"
#       "models--google--gemma-2b"  -> "google-gemma-2b"
#     """
#     base = os.path.basename(model_or_dir.rstrip("/"))
#     if base.startswith("models--"):
#         base = base[len("models--"):]
#     base = base.replace("--", "-").replace("/", "-")
#     return base

# # -----------------------
# # 读取三种方法的概念向量，并按层求均值（去掉 bias）
# # 只加载以 file_prefix 开头的文件，避免维度不匹配
# # 返回：{method: [tensor(v_l), ...]}
# # -----------------------
# def load_concept_vectors(samp_dir, device, dtype, file_prefix=None):
#     all_files = os.listdir(samp_dir)
#     if file_prefix:
#         all_files = [f for f in all_files if f.startswith(file_prefix + "_")]

#     methods = {
#         "diag":  sort_by_layer([f for f in all_files if f.endswith("_diag_samp.npy")]),
#         "full":  sort_by_layer([f for f in all_files if f.endswith("_full_samp.npy")]),
#         "stack": sort_by_layer([f for f in all_files if f.endswith("_stacking_samp.npy")]),
#     }
#     concept_vectors = {}
#     for m, files in methods.items():
#         vecs_per_layer = []
#         for fn in files:
#             arr = np.load(os.path.join(samp_dir, fn))   # (1000, d+1)
#             v = arr[:, :-1].mean(axis=0)                # 去 bias 后对 1000 条求均值 -> (d,)
#             t = torch.from_numpy(v).to(device)
#             t = t.to(dtype if dtype is not None else torch.float32)
#             vecs_per_layer.append(t)
#         if vecs_per_layer:
#             concept_vectors[m] = vecs_per_layer
#     return concept_vectors

# # -----------------------
# # 钩子：对“最后一个 token”的隐状态做凸组合
# # h' = (1-a) h + a * v_c（可做简单尺度匹配）
# # -----------------------
# def register_steering_hooks(model, concept_vecs, total_layers, alpha, scale_match=True):
#     # 找到层列表（不同模型命名不同）
#     if   hasattr(model, "transformer"): base = model.transformer
#     elif hasattr(model, "model"):       base = model.model
#     elif hasattr(model, "gemma"):       base = model.gemma
#     else:                               base = model

#     if   hasattr(base, "layers"): layers = base.layers
#     elif hasattr(base, "h"):      layers = base.h
#     elif hasattr(base, "decoder") and hasattr(base.decoder, "layers"):
#         layers = base.decoder.layers
#     else:
#         raise RuntimeError("找不到 Transformer 层列表，无法注册前向钩子。")

#     usable_layers = min(len(layers), len(concept_vecs), total_layers)
#     hooks = []

#     def make_hook(idx):
#         def hook(module, inputs, output):
#             if output is None or not (0 < idx < usable_layers - 1):
#                 return output

#             if isinstance(output, tuple):
#                 hidden, others = output[0], output[1:]
#             else:
#                 hidden, others = output, None

#             # hidden: (bsz, seq, dim)
#             h_last = hidden[:, -1, :]
#             cv = concept_vecs[idx].to(hidden.dtype).to(h_last.device)

#             # 维度检查（防止误加载到别的模型的向量）
#             if cv.numel() != h_last.shape[-1]:
#                 raise RuntimeError(
#                     f"[维度不匹配] layer {idx}: hidden_dim={h_last.shape[-1]}, concept_dim={cv.numel()}.\n"
#                     f"请确认 samp_dir 中文件前缀与当前模型一致，或用 --file_prefix 正确过滤。"
#                 )

#             if scale_match:
#                 eps = 1e-8
#                 scale = h_last.abs().mean() / (cv.abs().mean() + eps)
#                 cv = cv * scale

#             hidden[:, -1, :] = h_last * (1 - alpha) + cv * alpha

#             if others is None:
#                 return hidden
#             return (hidden,) + others
#         return hook

#     for i, layer in enumerate(layers[:usable_layers]):
#         if 0 < i < usable_layers - 1:
#             hooks.append(layer.register_forward_hook(make_hook(i)))
#     return hooks

# # -----------------------
# # 生成一次：若 concept_vecs 为 None 或 alpha=0，则生成原始输出
# # 显式传入 pad/eos（关键修复）
# # -----------------------
# def generate_once(model, tokenizer, prompt, device, max_new_tokens=80,
#                   concept_vecs=None, alpha=0.0, total_layers=None, scale_match=True):
#     inputs = tokenizer(prompt, return_tensors="pt")
#     input_ids = inputs["input_ids"].to(device)
#     attn = inputs.get("attention_mask")
#     attn = attn.to(device) if attn is not None else None

#     hooks = []
#     if concept_vecs is not None and alpha != 0.0:
#         hooks = register_steering_hooks(model, concept_vecs, total_layers, alpha, scale_match)

#     try:
#         out = model.generate(
#             input_ids=input_ids,
#             attention_mask=attn,
#             max_new_tokens=max_new_tokens,
#             pad_token_id=tokenizer.pad_token_id,
#             eos_token_id=tokenizer.eos_token_id,
#             do_sample=False,          # 需要可重复结果的话保持 greedy
#             temperature=None
#         )
#     finally:
#         for h in hooks:
#             h.remove()

#     gen_ids = out[0][input_ids.shape[1]:]
#     return tokenizer.decode(gen_ids, skip_special_tokens=True)

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model", default="google/gemma-2b", type=str,
#                         help="远程模型名（如 google/gemma-2b），当 --model_dir 为空时生效")
#     parser.add_argument("--model_dir", default="", type=str,
#                         help="本地模型目录；若给了，则优先生效（目录下需有 config.json 等）")
#     parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"],
#                         help="加载模型精度；很多 GPU 不支持 bf16，默认 fp16 更稳")
#     parser.add_argument("--samp_dir", default="lab_rs/samp", type=str, help="存放 *_samp.npy 的目录")
#     parser.add_argument("--file_prefix", default="", type=str,
#                         help="仅加载以该前缀开头的采样文件（例如 google-gemma-2b），不填则按模型名自动推断")
#     parser.add_argument("--prompts_file", default="", type=str, help="纯文本文件；每行一个 prompt。为空则用内置示例")
#     parser.add_argument("--alphas", default="0.1,0.3,0.5", type=str, help="逗号分隔的强度列表（可负数）")
#     parser.add_argument("--max_new_tokens", default=80, type=int)
#     parser.add_argument("--no_scale_match", action="store_true", help="关闭尺度匹配（默认开启）")
#     parser.add_argument("--out_csv", default="steering_results.csv", type=str)
#     args = parser.parse_args()

#     # 解析模型路径
#     model_path = args.model_dir if args.model_dir else args.model

#     # 选择 dtype
#     dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
#     torch_dtype = dtype_map.get(args.dtype, torch.float16)

#     # 1) 加载分词器与模型
#     tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

#     model = AutoModelForCausalLM.from_pretrained(
#         model_path,
#         device_map="auto",
#         torch_dtype=torch_dtype
#     )
#     model.eval()
#     device = next(model.parameters()).device
#     dtype = next(model.parameters()).dtype

#     # —— 关键：设置 pad/eos，避免 generate() 内部张量非法 —— #
#     if tokenizer.pad_token_id is None:
#         tokenizer.pad_token = tokenizer.eos_token
#     model.config.pad_token_id = tokenizer.pad_token_id
#     if getattr(model, "generation_config", None) is not None:
#         model.generation_config.pad_token_id = tokenizer.pad_token_id
#         model.generation_config.eos_token_id = tokenizer.eos_token_id

#     # 模型层数
#     if hasattr(model.config, "num_hidden_layers"):
#         total_layers = model.config.num_hidden_layers
#     elif hasattr(model.config, "n_layer"):
#         total_layers = model.config.n_layer
#     else:
#         total_layers = None  # 稍后用向量层数兜底

#     # 2) 读取向量（只加载前缀匹配的）
#     file_prefix = args.file_prefix.strip() or to_model_tag(model_path)
#     concept_vectors = load_concept_vectors(args.samp_dir, device, dtype, file_prefix=file_prefix)
#     if not concept_vectors:
#         raise RuntimeError(
#             f"在 {args.samp_dir} 未找到以 '{file_prefix}_' 开头的 *_samp.npy 文件。\n"
#             f"请检查 --file_prefix 或样本文件命名。"
#         )

#     # 若模型未能提供层数，用向量层数兜底
#     if total_layers is None:
#         any_method = next(iter(concept_vectors))
#         total_layers = len(concept_vectors[any_method])

#     # 3) 准备 prompts
#     if args.prompts_file and os.path.exists(args.prompts_file):
#         with open(args.prompts_file, "r", encoding="utf-8") as f:
#             prompts = [ln.strip() for ln in f if ln.strip()]
#     else:
#         prompts = [
#             # 内置一个与 STSA 情感相关的示例 prompt（负向改写）
#             ('You are a blunt film critic. Rewrite the following movie review into ONE short, clearly '
#              'negative sentence (12–20 words). Be direct. No emojis, no hashtags, no ratings, no quotes. '
#              'Sentence: "it has more than a few moments that are insightful enough to be fondly remembered '
#              'in the endlessly challenging maze of moviegoing." Negative review:')
#         ]

#     alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
#     scale_match = not args.no_scale_match

#     # 4) 逐条生成并写 CSV
#     os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
#     with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
#         wr = csv.writer(f)
#         wr.writerow(["method", "alpha", "prompt", "original_output", "steered_output"])

#         for method, vecs in concept_vectors.items():
#             for p in prompts:
#                 # 原始输出（不加 steering）
#                 orig = generate_once(
#                     model, tokenizer, p, device, args.max_new_tokens,
#                     concept_vecs=None, alpha=0.0, total_layers=total_layers, scale_match=scale_match
#                 )
#                 for a in alphas:
#                     steered = generate_once(
#                         model, tokenizer, p, device, args.max_new_tokens,
#                         concept_vecs=vecs, alpha=a, total_layers=total_layers, scale_match=scale_match
#                     )
#                     wr.writerow([method, a, p, orig, steered])
#                     print(f"[{method} α={a}] {p}\n  ├─ Orig: {orig}\n  └─ Steer: {steered}\n")

# if __name__ == "__main__":
#     main()
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-






# """
# Steering_generate.py
# 功能：按论文式(11)进行 activation steering，只负责“生成与落盘”，不做 GPT 评估
# 核心：剥离原文，只把指令编码进模型；手写 Greedy Decode，支持多种 steering method
# 输出：CSV，列为 [method, alpha, source, steered_output]
# """

# import os
# import re
# import csv
# import argparse
# import logging
# from typing import Dict, List, Optional

# import numpy as np
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM

# # —— 日志配置 —— #
# logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
# logger = logging.getLogger(__name__)

# # —— 解析层号 —— #
# LAYER_RE = re.compile(r"layer(\d+)", re.IGNORECASE)
# def parse_layer(fn: str) -> Optional[int]:
#     m = LAYER_RE.search(fn)
#     return int(m.group(1)) if m else None

# # —— 从文件名前缀猜数据集名 —— #
# def guess_dataset(prefix: str) -> str:
#     parts = prefix.split("_")
#     for p in parts:
#         if p.isupper() or p[0].isupper():
#             return p
#     return "UNKNOWN"

# # —— 读取概念向量（按 layer 对齐） —— #
# def load_vectors(samp_dir: str, device, dtype, prefix: str) -> Dict[str, Dict[int, torch.Tensor]]:
#     files = os.listdir(samp_dir)
#     if prefix:
#         files = [f for f in files if f.startswith(prefix + "_")]
#     methods = {
#         "diag":  [f for f in files if f.endswith("_diag_samp.npy")],
#         "full":  [f for f in files if f.endswith("_full_samp.npy")],
#         "stack": [f for f in files if f.endswith("_stacking_samp.npy")],
#     }
#     out: Dict[str, Dict[int, torch.Tensor]] = {}
#     for m, fl in methods.items():
#         d: Dict[int, torch.Tensor] = {}
#         for fn in fl:
#             l = parse_layer(fn)
#             if l is None: continue
#             arr = np.load(os.path.join(samp_dir, fn))  # (1000, d+1)
#             vec = arr[:, :-1].mean(axis=0)            # 去 bias, 求均值
#             t = torch.from_numpy(vec).to(device).to(dtype)
#             d[l] = t
#         if d:
#             out[m] = d
#     return out

# # —— 解析 layers_policy —— #
# def resolve_layers(total: int, policy: str, arg: str) -> List[int]:
#     if policy == "penultimate":
#         return [total-2] if total>=2 else []
#     if policy == "all":
#         return list(range(1, total-1))
#     if policy == "list":
#         return sorted(int(x) for x in arg.split(",") if x.strip().isdigit())
#     if policy == "range":
#         a,b = arg.split(":")
#         lo,hi = int(a),int(b)
#         return list(range(max(1,lo), min(total-1,hi)))
#     raise ValueError(f"Unknown layers_policy {policy}")

# # —— 注册 steering hook —— #
# def register_hooks(
#     model, vecs: Dict[int,torch.Tensor], layers: List[int],
#     alpha: float, direction: int, unit_norm: bool, scale: bool
# ) -> List[torch.utils.hooks.RemovableHandle]:
#     base = getattr(model, "transformer", getattr(model, "model",
#             getattr(model, "gemma", model)))
#     mods = getattr(base, "layers", getattr(base, "h",
#             getattr(getattr(base, "decoder",None),"layers",None)))
#     handles = []
#     def make_hook(idx: int):
#         def hook(_, __, out):
#             if idx not in vecs: return out
#             h = out[0] if isinstance(out, tuple) else out  # (bsz,seq,d)
#             last = h[:,-1,:]
#             cv = vecs[idx].to(last.device).to(last.dtype)
#             if unit_norm:
#                 cv = cv/(cv.norm(p=2)+1e-8)
#             if direction!=0:
#                 cv = cv*direction
#             if scale:
#                 cv = cv*(last.abs().mean()/(cv.abs().mean()+1e-8))
#             h[:,-1,:] = last*(1-alpha)+cv*alpha
#             return (h,)+out[1:] if isinstance(out, tuple) else h
#         return hook
#     for idx in layers:
#         if idx < len(mods):
#             handles.append(mods[idx].register_forward_hook(make_hook(idx)))
#     return handles

# # —— 手写 Greedy Decode —— #
# def greedy_decode(model, tokenizer, ids, max_new, hooks):
#     seq = ids
#     with torch.no_grad():
#         for _ in range(max_new):
#             mask = torch.ones_like(seq, dtype=torch.long, device=seq.device)
#             out = model(input_ids=seq, attention_mask=mask, use_cache=False)
#             logits = out.logits  # (1,seq,vocab)
#             nt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)  # (1,1)
#             seq = torch.cat([seq, nt], dim=-1)
#             if nt.item()==tokenizer.eos_token_id:
#                 break
#     for h in hooks: h.remove()
#     return seq[:, ids.shape[1]:]

# # —— 生成一次 —— #
# def generate_once(
#     model, tokenizer,
#     instr: str,  # 只包含指令，不带原文
#     device, max_new,
#     vecs: Optional[Dict[int,torch.Tensor]],
#     layers: List[int],
#     alpha: float, direction: int,
#     unit_norm: bool, scale: bool
# ) -> str:
#     enc = tokenizer(instr, return_tensors="pt")
#     ids = enc["input_ids"].to(device)
#     hooks = []
#     if vecs and abs(alpha)>0 and layers:
#         dirn = direction if direction!=0 else (1 if alpha>0 else -1)
#         hooks = register_hooks(model, vecs, layers, abs(alpha), dirn, unit_norm, scale)
#     out_seq = greedy_decode(model, tokenizer, ids, max_new, hooks)
#     return tokenizer.decode(out_seq[0], skip_special_tokens=True)

# # —— 主逻辑 —— #
# def main():
#     p = argparse.ArgumentParser()
#     p.add_argument("--model",     default="google/gemma-2b")
#     p.add_argument("--model_dir", default="", help="本地模型目录")
#     p.add_argument("--dtype",     default="fp32", choices=["fp16","bf16","fp32"])
#     p.add_argument("--samp_dir",  default="lab_rs/samp")
#     p.add_argument("--prefix",    default="",     help="采样文件前缀")
#     p.add_argument("--layers_policy", default="penultimate",
#                    choices=["penultimate","all","list","range"])
#     p.add_argument("--layers",    default="", help="list: '5,12' 或 range: '5:12'")
#     p.add_argument("--alphas",    default="0.04,0.06")
#     p.add_argument("--direction", type=int, default=0,
#                    help="方向+1/-1,0=由alpha符号决定")
#     p.add_argument("--unit_norm", action="store_true")
#     p.add_argument("--no_scale",  action="store_true")
#     p.add_argument("--max_new",   type=int, default=60)
#     p.add_argument("--source_file", default="", help="每行一个 source (原文句子)")
#     p.add_argument("--out_csv",   default="steering_results.csv")
#     args = p.parse_args()

#     mp = args.model_dir or args.model
#     dt = {"fp16":torch.float16,"bf16":torch.bfloat16,"fp32":torch.float32}[args.dtype]

#     tok = AutoTokenizer.from_pretrained(mp, use_fast=False)
#     model = AutoModelForCausalLM.from_pretrained(mp, device_map="auto", torch_dtype=dt)
#     model.eval()
#     dev = next(model.parameters()).device

#     # pad_token
#     if tok.pad_token_id is None:
#         tok.pad_token = tok.eos_token
#     model.config.pad_token_id = tok.pad_token_id
#     if hasattr(model, "generation_config"):
#         model.generation_config.pad_token_id = tok.pad_token_id
#         model.generation_config.eos_token_id  = tok.eos_token_id

#     total = getattr(model.config, "num_hidden_layers", None) or getattr(model.config, "n_layer",0)
#     if total<=1:
#         raise RuntimeError("层数太少")

#     prefix = args.prefix or mp.replace("/","-")
#     ds = guess_dataset(prefix)
#     cvs = load_vectors(args.samp_dir, dev, dt, prefix)
#     if not cvs:
#         raise RuntimeError("找不到向量")

#     layers = resolve_layers(total, args.layers_policy, args.layers)
#     # 只保留存在向量的层
#     for m in list(cvs):
#         cvs[m] = {l:cvs[m][l] for l in layers if l in cvs[m]}

#     logger.info(f"Model={mp} Dataset={ds} Layers={layers}")
#     logger.info(f"Methods={list(cvs.keys())} Alphas={args.alphas}")
#     logger.info(f"unit_norm={args.unit_norm} scale_match={args.no_scale}")

#     # 加载所有 source
#     sources: List[str] = []
#     if args.source_file and os.path.exists(args.source_file):
#         with open(args.source_file,"r",encoding="utf8") as f:
#             sources = [ln.strip() for ln in f if ln.strip()]
#     else:
#                 # -*- coding: utf-8 -*-
#         # 带标签的句子列表，格式为 (label, sentence)
#         sources = [
#             ("this is one of polanski 's best films ."),
#             ("take care of my cat offers a refreshingly different slice of asian cinema ."),
#             ("acting , particularly by tambor , almost makes `` never again '' worthwhile , but -lrb- writer/director -rrb- schaeffer should follow his titular advice"),
#             ("the movie exists for its soccer action and its fine acting ."),
#             ("arnold 's jump from little screen to big will leave frowns on more than a few faces ."),
#             ("if this holiday movie is supposed to be a gift , somebody unwrapped it early , took out all the good stuff , and left behind the crap -lrb- literally -rrb- ."),
#             ("jason x has cheesy effects and a hoary plot , but its macabre , self-deprecating sense of humor makes up for a lot ."),
#             ("even as lame horror flicks go , this is lame ."),
#             ("oft-described as the antidote to american pie-type sex comedies , it actually has a bundle in common with them , as the film diffuses every opportunity for a breakthrough"),
#             ("though the violence is far less sadistic than usual , the film is typical miike : fast , furious and full of off-the-cuff imaginative flourishes ."),
#             ("when a set of pre-shooting guidelines a director came up with for his actors turns out to be cleverer , better written and of considerable more interest than the finished film , that's a bad sign ."),
#             ("the passions aroused by the discord between old and new cultures are set against the strange , stark beauty of the mideast desert , so lovingly and perceptively filmed that you can almost taste the desiccated air ."),
#             ("if your senses have n't been dulled by slasher films and gorefests , if you're a connoisseur of psychological horror , this is your ticket ."),
#             ("any one episode of the sopranos would send this ill-conceived folly to sleep with the fishes ."),
#             ("as conceived by mr. schaeffer , christopher and grace are little more than collections of quirky traits lifted from a screenwriter 's outline and thrown at actors charged with the impossible task of making them jell ."),
#             ("those who managed to avoid the deconstructionist theorizing of french philosopher jacques derrida in college can now take an 85-minute brush-up course with the documentary derrida ."),
#             ("most new movies have a bright sheen ."),
#         ]


#     # 指令模板
#     instr = (
#         "You are a blunt film critic. Rewrite the following movie review into "
#         "ONE short, clearly sentence (12–20 words). Be direct. "
#         "No emojis, no hashtags, no ratings, no quotes. Negative review:"
#     )

#     alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
#     unit_norm = bool(args.unit_norm)
#     scale = not args.no_scale

#     os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
#     with open(args.out_csv, "w", newline="", encoding="utf8") as fout:
#         writer = csv.writer(fout)
#         writer.writerow(["method","alpha","source","steered_output"])

#         for method, vecs in cvs.items():
#             logger.info(f"=== Method={method} ===")
#             for src in sources:
#                 for a in alphas:
#                     out = generate_once(
#                         model, tok, instr, dev, args.max_new,
#                         vecs, layers, a, args.direction,
#                         unit_norm, scale
#                     )
#                     writer.writerow([method, a, src, out])
#                     logger.info(f"[{method}@{a}] src=… out={out[:80]}...")

# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
# steering_generate_from_main.py
# 需求对齐：数据/模型加载与 main.py 一致；使用 stacking 生成的 concept vectors 做 Eq.(11) steering；导出 CSV

# import argparse
# import os
# import numpy as np
# import torch
# import csv
# import pickle
# import logging
# from tqdm import tqdm
# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# # 你的工程内模块（与 main.py 相同）
# from dataset import DataProcessing
# from util import LLM  # 这里只为保持一致性导入，不强制使用 LLM 生成

# logger = logging.getLogger(__name__)

# def build_model_and_tokenizer(model_name, cache_dir, quant, cuda):
#     quant_cfg = None
#     if quant == 8:
#         quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
#     tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
#     if quant == 32:
#         model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
#     elif quant == 8:
#         model = AutoModelForCausalLM.from_pretrained(
#             model_name, quantization_config=quant_cfg, cache_dir=cache_dir
#         )
#     else:  # 16
#         model = AutoModelForCausalLM.from_pretrained(
#             model_name, torch_dtype=torch.bfloat16, cache_dir=cache_dir
#         )
#     device = torch.device(f"cuda:{cuda}" if (cuda >= 0 and torch.cuda.is_available()) else "cpu")
#     model.to(device)
#     model.eval()
#     return model, tokenizer, device

# def infer_layer_count(model_name, model):
#     layer_map = {
#         "google/gemma-2b":18, "google/gemma-7b":28,
#         "meta-llama/Llama-2-7b-chat-hf":32, "meta-llama/Llama-2-13b-chat-hf":40,
#         "meta-llama/Llama-2-70b-chat-hf":80,
#         "Qwen/Qwen1.5-0.5B":24, "Qwen/Qwen1.5-1.8B":24,
#         "Qwen/Qwen1.5-4B":40,   "Qwen/Qwen1.5-7B":32,
#         "Qwen/Qwen1.5-14B":40,  "Qwen/Qwen1.5-72B":80
#     }
#     if model_name in layer_map:
#         return layer_map[model_name]
#     # 回退：从 config 读取
#     if hasattr(model.config, "num_hidden_layers"):
#         return model.config.num_hidden_layers
#     if hasattr(model.config, "n_layer"):
#         return model.config.n_layer
#     raise ValueError("无法确定模型层数，请在 layer_map 中补充该模型。")

# def load_stack_vectors(savepath, model_name, dataset_tag, tot_layer, device):
#     base = f"{model_name.replace('/','-')}_{dataset_tag}"
#     samp_dir = os.path.join(savepath, "samp")
#     vectors = []
#     for l in range(tot_layer):
#         f = os.path.join(samp_dir, f"{base}_layer{l}_stacking_samp.npy")
#         if not os.path.isfile(f):
#             raise FileNotFoundError(f"未找到 stacking 概念向量：{f}")
#         arr = np.load(f)  # (1, d+1)
#         if arr.ndim != 2 or arr.shape[0] != 1:
#             raise ValueError(f"{f} 形状应为 (1, d+1)，实际 {arr.shape}")
#         w = arr[0, :-1].astype(np.float32)  # 只取 w，丢弃 b
#         vectors.append(torch.tensor(w, device=device))
#     return vectors  # list[torch.Tensor], 每层一个 (d,)

# def get_blocks(model):
#     # 适配常见结构（Gemma/LLaMA/Qwen等）
#     if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
#         return model.transformer.h
#     if hasattr(model, 'model') and hasattr(model.model, 'transformer') and hasattr(model.model.transformer, 'h'):
#         return model.model.transformer.h
#     if hasattr(model, 'model') and hasattr(model.model, 'layers'):
#         return model.model.layers
#     raise RuntimeError("无法定位 transformer blocks，请按实际模型结构修改 get_blocks。")

# def register_steering_hooks(model, vectors, a, sign, tot_layer, single_layer=-1):
#     """在中间层（默认除首末层）对最后一个 token 的隐藏状态注入：h := h + sign * a * v_l"""
#     blocks = get_blocks(model)
#     handles = []

#     def make_hook(v):
#         def hook_fn(module, inputs, output):
#             # output 可能是 Tensor 或 (Tensor, *others)
#             if isinstance(output, tuple):
#                 hidden = output[0]
#                 others = output[1:]
#             else:
#                 hidden = output
#                 others = None
#             # 只改最后一个 token
#             last = hidden[:, -1, :]  # (B, d)
#             if v.shape[-1] != last.shape[-1]:
#                 raise ValueError(f"概念向量维度 {v.shape[-1]} 与隐藏维 {last.shape[-1]} 不符")
#             new_last = last + (sign * a) * v
#             hidden = hidden.clone()
#             hidden[:, -1, :] = new_last
#             if others is None:
#                 return hidden
#             else:
#                 return (hidden, *others)
#         return hook_fn

#     for l, block in enumerate(blocks):
#         if l == 0 or l == tot_layer - 1:
#             continue  # 跳过首/末层
#         if single_layer != -1 and l != single_layer:
#             continue
#         h = block.register_forward_hook(make_hook(vectors[l]))
#         handles.append(h)

#     return handles

# def greedy_generate_with_hooks(model, tokenizer, device, prompt_ids, max_new_tokens=20):
#     # 以 use_cache 流水式生成，逐 token greedy
#     input_ids = prompt_ids.to(device)
#     attention_mask = torch.ones_like(input_ids, device=device)
#     # 先喂除最后一个 token，保留最后一个 token 单独走（便于 hooks 在“last token”上命中）
#     if input_ids.shape[1] > 1:
#         out = model(input_ids=input_ids[:, :-1],
#                     attention_mask=attention_mask[:, :-1],
#                     use_cache=True)
#         past = out.past_key_values
#         cur = input_ids[:, -1:]
#     else:
#         past = None
#         cur = input_ids
#     gen_tokens = []
#     with torch.no_grad():
#         for _ in range(max_new_tokens):
#             out = model(input_ids=cur, past_key_values=past, use_cache=True)
#             logits = out.logits[:, -1, :]
#             past = out.past_key_values
#             next_id = torch.argmax(logits, dim=-1)  # greedy
#             if tokenizer.eos_token_id is not None and next_id.item() == tokenizer.eos_token_id:
#                 break
#             gen_tokens.append(next_id.item())
#             cur = next_id.unsqueeze(0)
#     return tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

# def main():
#     parser = argparse.ArgumentParser()
#     # —— 与 main.py 对齐的核心参数 —— #
#     parser.add_argument('--savepath',   type=str, default="./lab_rs/", help="结果保存目录(读取 stacking 概念向量)")
#     parser.add_argument('--model_path', type=str, default="./",        help="模型缓存目录")
#     parser.add_argument('--dataset',    type=str, default='STSA',      help="数据集名称（用于文件名 tag）")
#     parser.add_argument('--datapath',   type=str, default='./dataset/stsa.binary.train', help="数据文件路径")
#     parser.add_argument('--model',      type=str, default='google/gemma-2b', help="LLM 模型")
#     parser.add_argument('--cuda',       type=int, default=0,           help="CUDA 设备号；CPU 用 -1")
#     parser.add_argument('--quant',      type=int, default=16, choices=[8,16,32], help="量化位数")
#     parser.add_argument('--noise',      type=str, default='non-noise', help="与 DataProcessing 保持一致")
#     parser.add_argument('--concept',    type=str, default='', help='从 dataset/raw/{concept}.pkl 读数据')

#     # —— steering & 运行控制 —— #
#     parser.add_argument('--alpha',      type=float, default=0.064, help='steering 强度 a（论文建议扫多个值）')
#     parser.add_argument('--max_tokens', type=int, default=20,      help='生成的新 token 数')
#     parser.add_argument('--single_layer', type=int, default=-1,    help='只在该层施加；-1 表示所有中间层')
#     parser.add_argument('--output',     type=str, default='./results/steer_from_main.csv', help='CSV 输出路径')
#     args = parser.parse_args()

#     os.makedirs(os.path.dirname(args.output), exist_ok=True)

#     # —— 模型加载（完全与 main.py 同风格） —— #
#     model, tokenizer, device = build_model_and_tokenizer(
#         args.model, args.model_path, args.quant, args.cuda
#     )
#     tot_layer = infer_layer_count(args.model, model)

#     # —— 数据读取（与 main.py 一致的两分支） —— #
#     if args.concept:
#         one_pkl = os.path.join('dataset', 'raw', f"{args.concept}.pkl")
#         if not os.path.isfile(one_pkl):
#             raise FileNotFoundError(f"{one_pkl} 不存在，请检查概念名")
#         with open(one_pkl, 'rb') as f:
#             d = pickle.load(f)
#         key = list(d.keys())[0]
#         pos_q = d[key].get('positive', [])
#         neg_q = d[key].get('negative', [])
#         dataset_tag = args.concept
#         # 你 main 里 prompt_tmpl/cot 主要用于取隐表示；这里我们直接用原文作上下文
#         prompt_tmpl = "{text}"
#         cot = False
#         print(f"✅ 概念 {args.concept}: 正例 {len(pos_q)} | 负例 {len(neg_q)}")
#     else:
#         DP = DataProcessing(data_path=args.datapath, data_name=args.dataset, noise=args.noise)
#         pos_q, neg_q, prompt_tmpl, cot = DP.dispacher()
#         dataset_tag = args.dataset
#         print(f"✅ DataProcessing 读取: 正例 {len(pos_q)} | 负例 {len(neg_q)} | dataset_tag={dataset_tag}")

#     # —— 概念向量（stacking）加载 —— #
#     v_layers = load_stack_vectors(args.savepath, args.model, dataset_tag, tot_layer, device)

#     # —— 构造中性生成模板：避免在文本上直接给出正/负指令，以便让注入起主导作用 —— #
#     def build_prompt(text):
#         # 统一的中性提示：给出原句上下文，请模型写一句“感受”
#         return f"Review:\n{text}\nOne-sentence reaction:"

#     # —— CSV 输出 —— #
#     with open(args.output, 'w', newline='', encoding='utf-8') as f:
#         wr = csv.writer(f)
#         wr.writerow([
#             "model", "layers", "alpha",
#             "orig_label", "target_label",
#             "original_text", "steered_output"
#         ])

#         # ====== 方向一：positive → negative（沿用 w 的正方向表示“正类”，因此取 sign = -1） ======
#         for q in tqdm(pos_q, desc="Steer POS→NEG"):
#             text = q #if args.concept else (build_prompt(q))  # 若 concept 模式直接文本，否则包模板
#             # 编码
#             enc = tokenizer(text, return_tensors='pt')
#             prompt_ids = enc['input_ids']
#             # 注册 hooks
#             handles = register_steering_hooks(
#                 model, v_layers, a=args.alpha, sign=-1.0, tot_layer=tot_layer, single_layer=args.single_layer
#             )
#             # 生成
#             steered = greedy_generate_with_hooks(
#                 model, tokenizer, device, prompt_ids, max_new_tokens=args.max_tokens
#             )
#             # 移除 hooks
#             for h in handles: h.remove()
#             # 写 CSV
#             layer_tag = "ALL-MID" if args.single_layer == -1 else str(args.single_layer)
#             wr.writerow([args.model, layer_tag, args.alpha, "positive", "negative", q, steered])

#         # ====== 方向二：negative → positive（sign = +1） ======
#         for q in tqdm(neg_q, desc="Steer NEG→POS"):
#             text = q if args.concept else (build_prompt(q))
#             enc = tokenizer(text, return_tensors='pt')
#             prompt_ids = enc['input_ids']
#             handles = register_steering_hooks(
#                 model, v_layers, a=args.alpha, sign=+1.0, tot_layer=tot_layer, single_layer=args.single_layer
#             )
#             steered = greedy_generate_with_hooks(
#                 model, tokenizer, device, prompt_ids, max_new_tokens=args.max_tokens
#             )
#             for h in handles: h.remove()
#             layer_tag = "ALL-MID" if args.single_layer == -1 else str(args.single_layer)
#             wr.writerow([args.model, layer_tag, args.alpha, "negative", "positive", q, steered])

#     print(f"✅ Done. CSV saved to: {args.output}")

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-layer steering generation:
- Only ONE layer is steered per run (single-layer injection)
- Iterate all middle layers (exclude first & last) -> 16 layers for Gemma-2B
- For each layer: run two directions
    * pos→neg: use a negative-review prompt, steer with -w
    * neg→pos: use a positive-review prompt, steer with +w
- Each direction generates N samples (default 100) with the SAME prompt (sampling)
- Save CSV with layer id, direction, alpha, etc.

Concept vectors expected at:
    {savepath}/samp/{model_tag}_{dataset_tag}_layer{L}_stacking_samp.npy   (store [w|b], use w)

Example:
    python Steering_generate.py \
        --savepath ./lab_rs \
        --model google/gemma-2b --model_path ./ --quant 16 --cuda 0 \
        --dataset_tag STSA \
        --alpha 0.064 \
        --num_samples 100 \
        --max_tokens 80 \
        --out_csv ./results/gemma2b_STSA_perlayer_steer.csv
"""

import os, csv, argparse, random
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


# -----------------------
# Utilities
# -----------------------
def set_seed_local(seed: int):
    """Set RNG seeds for reproducibility (no generator kwarg needed)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------
# Model helpers
# -----------------------
def load_model_and_tokenizer(model_name, cache_dir, quant, cuda):
    qcfg = None
    if quant == 8:
        qcfg = BitsAndBytesConfig(load_in_8bit=True)
    tok = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    if quant == 32:
        mdl = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
    elif quant == 8:
        mdl = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=qcfg, cache_dir=cache_dir)
    else:  # 16
        mdl = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, cache_dir=cache_dir)
    device = torch.device(f"cuda:{cuda}" if (cuda >= 0 and torch.cuda.is_available()) else "cpu")
    mdl.to(device).eval()
    return mdl, tok, device


def infer_num_layers(model_name, model):
    layer_map = {
        "google/gemma-2b": 18, "google/gemma-7b": 28,
        "meta-llama/Llama-2-7b-chat-hf": 32, "meta-llama/Llama-2-13b-chat-hf": 40,
        "meta-llama/Llama-2-70b-chat-hf": 80,
        "Qwen/Qwen1.5-0.5B": 24, "Qwen/Qwen1.5-1.8B": 24,
        "Qwen/Qwen1.5-4B": 40, "Qwen/Qwen1.5-7B": 32,
        "Qwen/Qwen1.5-14B": 40, "Qwen/Qwen1.5-72B": 80
    }
    if model_name in layer_map:
        return layer_map[model_name]
    if hasattr(model.config, "num_hidden_layers"):  # HF common
        return int(model.config.num_hidden_layers)
    if hasattr(model.config, "n_layer"):            # some variants
        return int(model.config.n_layer)
    raise ValueError("无法确定模型层数，请在 layer_map 中补充该模型。")


def get_blocks(model):
    # Try common layouts (Gemma/LLaMA/Qwen families)
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h
    if hasattr(model, 'model') and hasattr(model.model, 'transformer') and hasattr(model.model.transformer, 'h'):
        return model.model.transformer.h
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    raise RuntimeError("无法定位 transformer blocks。")


# -----------------------
# Concept vectors
# -----------------------
def load_concept_vectors(savepath, model_name, dataset_tag, tot_layer, device):
    """
    Load per-layer stacking vectors:
        {savepath}/samp/{base}_layer{L}_stacking_samp.npy  -> array (1, d+1)
    We take 'w' ([:-1]) as steering vector for that layer.
    """
    base = f"{model_name.replace('/','-')}_{dataset_tag}"
    samp_dir = os.path.join(savepath, "samp")
    vecs = []
    for L in range(tot_layer):
        f = os.path.join(samp_dir, f"{base}_layer{L}_stacking_samp.npy")
        if not os.path.isfile(f):
            vecs.append(None)  # for safety (will skip if not middle layer)
            continue
        arr = np.load(f)  # (1, d+1)
        if arr.ndim != 2 or arr.shape[0] != 1:
            raise ValueError(f"{f} 形状应为 (1, d+1)，实际 {arr.shape}")
        w = arr[0, :-1].astype(np.float32)      # drop bias, keep w
        vecs.append(torch.tensor(w, device=device))
    return vecs


# -----------------------
# Hook: single-layer injection
# -----------------------
def register_single_layer_hook(model, vectors, target_layer, alpha, sign, formula="add"):
    """
    Only steer ONE layer (target_layer). Others untouched.
    formula: "add" -> h := h + sign*alpha*v
             "convex" -> h := (1-alpha)*h + sign*alpha*v
    """
    blocks = get_blocks(model)
    handles = []

    if target_layer <= 0 or target_layer >= (len(blocks)-1):
        raise ValueError(f"只允许中间层，给定层 {target_layer} 无效。")

    v = vectors[target_layer]
    if v is None:
        raise FileNotFoundError(f"未找到该层的概念向量：layer{target_layer}")

    def hook_fn(module, inputs, output):
        if isinstance(output, tuple):
            hidden, others = output[0], output[1:]
        else:
            hidden, others = output, None
        last = hidden[:, -1, :]
        if v.shape[-1] != last.shape[-1]:
            raise ValueError(f"维度不匹配：v={v.shape[-1]} vs h={last.shape[-1]}")
        if formula == "convex":
            new_last = (1.0 - alpha) * last + (sign * alpha) * v
        else:
            new_last = last + (sign * alpha) * v
        hidden = hidden.clone()
        hidden[:, -1, :] = new_last
        return (hidden, *others) if others is not None else hidden

    h = blocks[target_layer].register_forward_hook(hook_fn)
    handles.append(h)
    return handles


# -----------------------
# Prompts (fixed; same prompt used N times with sampling)
# -----------------------
POS_PROMPT = (
    "You are a movie critic. Write exactly ONE short paragraph in English (<=60 words) "
    "giving a clearly positive movie review. No lists, no steps, no headings or labels, "
    "no prefaces. Start directly with the review sentence. No emojis."
)
NEG_PROMPT = (
    "You are a movie critic. Write exactly ONE short paragraph in English (<=60 words) "
    "giving a clearly negative movie review. Be candid but non-profane. No lists, no steps, "
    "no headings or labels, no prefaces. Start directly with the review sentence. No emojis."
)


# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--savepath',   type=str, default="./lab_rs")
    parser.add_argument('--model_path', type=str, default="./")
    parser.add_argument('--model',      type=str, default="google/gemma-2b")
    parser.add_argument('--quant',      type=int, default=16, choices=[8,16,32])
    parser.add_argument('--cuda',       type=int, default=0)
    parser.add_argument('--dataset_tag',type=str, default="STSA",
                        help="用于拼接概念向量文件名的标签（与你做 stacking 时一致，如 STSA）")

    parser.add_argument('--alpha',      type=float, default=0.064)
    parser.add_argument('--formula',    choices=['add','convex'], default='add',
                        help='注入公式：add = h + s*a*v；convex = (1-a)h + s*a*v')
    parser.add_argument('--num_samples',type=int, default=100,
                        help='每个方向（pos→neg / neg→pos）各生成多少条')
    parser.add_argument('--max_tokens', type=int, default=80,
                        help='最大生成 token 数（60英文词一般需要 ~70-90 token）')
    parser.add_argument('--top_p',      type=float, default=0.8)
    parser.add_argument('--temperature',type=float, default=0.5)
    parser.add_argument('--repetition_penalty', type=float, default=1.1)
    parser.add_argument('--no_repeat_ngram_size', type=int, default=3)

    parser.add_argument('--out_csv',    type=str, default="./results/perlayer_steer.csv")
    parser.add_argument('--seed',       type=int, default=42)

    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    # 1) Model & vectors
    model, tokenizer, device = load_model_and_tokenizer(args.model, args.model_path, args.quant, args.cuda)
    tot_layer = infer_num_layers(args.model, model)
    vectors   = load_concept_vectors(args.savepath, args.model, args.dataset_tag, tot_layer, device)

    # 2) Middle layers list (exclude first & last)
    middle_layers = list(range(1, tot_layer-1))  # Gemma-2B -> 1..16

    # 3) Setup decoding (sampling) & phrase filter（工程降噪，可去掉）
    bad_phrases = ["Answer:", "answer:", "Step", "Steps", "step", "steps", "Your review must", "your review must", "the following", "The following", "You may use", "you may use", "You can use", "you can use", "Title:", "Director:", "Stars:", "Plot:", "policy", "inappropriate content", "as an AI", "I cannot", "I can't", "I'm not sure", "i am not sure"]
    bad_words_ids = tokenizer(bad_phrases, add_special_tokens=False).input_ids

    # 4) CSV
    with open(args.out_csv, 'w', newline='', encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerow(["model","dataset_tag","alpha","formula","layer","direction","sample_id","prompt","generated_text"])

        # 固定 prompts
        pos_prompt = POS_PROMPT
        neg_prompt = NEG_PROMPT

        # 5) For each middle layer, run two directions; each generate N samples
        for L in middle_layers:
            # ---- Direction A: neg→pos (positive prompt; steer +w)
            enc = tokenizer(neg_prompt, return_tensors='pt')
            prompt_ids_pos = enc['input_ids'].to(device)

            handles = register_single_layer_hook(model, vectors, target_layer=L,
                                                 alpha=args.alpha, sign=+1.0, formula=args.formula)
            for i in tqdm(range(args.num_samples), desc=f"Layer {L:02d}  NEG→POS (+w)"):
                set_seed_local(args.seed + i + 12345)
                out_ids = model.generate(
                    input_ids=prompt_ids_pos,
                    max_new_tokens=args.max_tokens,
                    do_sample=True, top_p=args.top_p, temperature=args.temperature,
                    repetition_penalty=args.repetition_penalty,
                    no_repeat_ngram_size=args.no_repeat_ngram_size,
                    bad_words_ids=bad_words_ids,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
                txt = tokenizer.decode(out_ids[0, prompt_ids_pos.shape[1]:], skip_special_tokens=True).strip()
                wr.writerow([args.model, args.dataset_tag, args.alpha, args.formula, L, "neg→pos", i, neg_prompt, txt])
            for h in handles: h.remove()

            # ---- Direction B: pos→neg (negative prompt; steer -w)
            enc = tokenizer(pos_prompt, return_tensors='pt')
            prompt_ids_neg = enc['input_ids'].to(device)

            handles = register_single_layer_hook(model, vectors, target_layer=L,
                                                 alpha=args.alpha, sign=-1.0, formula=args.formula)
            for i in tqdm(range(args.num_samples), desc=f"Layer {L:02d}  POS→NEG (-w)"):
                set_seed_local(args.seed + i + 24680)
                out_ids = model.generate(
                    input_ids=prompt_ids_neg,
                    max_new_tokens=args.max_tokens,
                    do_sample=True, top_p=args.top_p, temperature=args.temperature,
                    repetition_penalty=args.repetition_penalty,
                    no_repeat_ngram_size=args.no_repeat_ngram_size,
                    bad_words_ids=bad_words_ids,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
                txt = tokenizer.decode(out_ids[0, prompt_ids_neg.shape[1]:], skip_special_tokens=True).strip()
                wr.writerow([args.model, args.dataset_tag, args.alpha, args.formula, L, "pos→neg", i, pos_prompt, txt])
            for h in handles: h.remove()

    print(f"✅ Done. Results saved to: {args.out_csv}")


if __name__ == "__main__":
    main()

