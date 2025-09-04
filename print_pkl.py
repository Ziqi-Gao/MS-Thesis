#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle

def load_all_pkls(raw_dir):
    """
    遍历 raw_dir 下的所有 .pkl 文件，加载并返回一个字典：
      {
        "Action": { "positive": [...], "negative": [...] },
        "Bird":   { ... },
        ...
      }
    """
    data = {}
    for fname in os.listdir(raw_dir):
        if not fname.endswith('.pkl'):
            continue
        concept = fname[:-4]  # 去掉 .pkl 后缀
        path = os.path.join(raw_dir, fname)
        try:
            with open(path, 'rb') as f:
                content = pickle.load(f)
            data[concept] = content
        except Exception as e:
            print(f"⚠️ 加载 {fname} 时出错：{e}")
    return data

def main():
    raw_dir = os.path.join('dataset', 'raw')  # 或者改成你实际路径
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(f"目录不存在：{raw_dir}")
    
    all_data = load_all_pkls(raw_dir)

    for concept_fname, content in all_data.items():
        real_key = next(iter(content))               # pkl 内实际概念名
        buckets = content[real_key]
        pos = len(buckets.get('positive', []))
        neg = len(buckets.get('negative', []))
        print(f"{real_key}: 正样本 {pos} 条，负样本 {neg} 条")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inspect_pkl.py

用法:
    python inspect_pkl.py <pkl_path> [--head N]

 - <pkl_path> : 目标 .pkl 文件路径
 - --head N   : 每类最多打印前 N 条示范句子 (默认 5)
"""
# import argparse, pickle, textwrap, pathlib, sys

# def main():
#     parser = argparse.ArgumentParser(description="查看 .pkl 样本文件内容")
#     parser.add_argument("pkl_path", type=str, help="待查看的 .pkl 文件路径")
#     parser.add_argument("--head", type=int, default=5, help="每类示例条数")
#     args = parser.parse_args()

#     pkl_path = pathlib.Path(args.pkl_path)
#     if not pkl_path.is_file():
#         sys.exit(f"❌ 找不到文件: {pkl_path}")

#     with pkl_path.open("rb") as f:
#         data = pickle.load(f)          # {concept: {"positive":[...], "negative":[...]}}
    
#     # 取第一层 key（通常只有 1 个概念）
#     for concept, buckets in data.items():
#         pos = buckets.get("positive", [])
#         neg = buckets.get("negative", [])
#         print(f"\n=== 概念: {concept} ===")
#         print(f"正样本: {len(pos)} 条 | 负样本: {len(neg)} 条\n")

#         # 打印前 N 条示范
#         head_n = args.head
#         print("- 正样本示例 -")
#         for i, sent in enumerate(pos[:head_n], 1):
#             print(textwrap.shorten(f"{i}. {sent}", width=120))
#         if len(pos) > head_n:
#             print(f"... (剩余 {len(pos)-head_n} 条省略)")

#         print("\n- 负样本示例 -")
#         for i, sent in enumerate(neg[:head_n], 1):
#             print(textwrap.shorten(f"{i}. {sent}", width=120))
#         if len(neg) > head_n:
#             print(f"... (剩余 {len(neg)-head_n} 条省略)")

# if __name__ == "__main__":
#     main()
