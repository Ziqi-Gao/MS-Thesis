#!/usr/bin/env python3
# 文件名：print_w.py
# 直接读取并打印 /home/del6500/CD/Concept.npz 中的 w

import numpy as np
import sys

def main():
    npz_path = '/gpfs/projects/p32737/del6500_home/CD/lab_rs/google_gemma-2b_cities_149_b.npz'
    try:
        data = np.load(npz_path)
    except Exception as e:
        print(f"无法打开文件 {npz_path}：{e}", file=sys.stderr)
        sys.exit(1)

    if "W" not in data.files:
        print(f"文件中不包含 'w'，可用的键有：{data.files}", file=sys.stderr)
        #sys.exit(1)

    w_all = data["B"]
    print(f"'w_all' 数组的形状：{w_all.shape}")
    print("内容如下：")
    print(w_all)

if __name__ == "__main__":
    main()
