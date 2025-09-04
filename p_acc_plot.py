# #!/usr/bin/env python3
# # plot_layered_acc.py
# # 功能：针对 lab_rs 下每组实验(prefix)，将原始 Acc、gauss_diag Acc、gauss_full Acc
# # 按层绘制分组柱状图，每层展示三种方法的 mean accuracy，图像保存在 lab_rs/plot。

# import os
# import glob
# import numpy as np
# import matplotlib.pyplot as plt

# LAB_RS_DIR = "./lab_rs"
# PLOT_DIR = os.path.join("./plot", "Acc")
# os.makedirs(PLOT_DIR, exist_ok=True)

# # 找到所有原始 acc 文件（排除 gauss_diag 和 gauss_full）
# orig_paths = glob.glob(os.path.join(LAB_RS_DIR, "*_acc.npz"))
# orig_paths = [p for p in orig_paths if ("gauss_diag" not in p and "gauss_full" not in p)]

# for orig_path in orig_paths:
#     prefix = os.path.basename(orig_path)[:-len("_acc.npz")]
#     diag_path = os.path.join(LAB_RS_DIR, f"{prefix}_gauss_diag_acc.npz")
#     full_path = os.path.join(LAB_RS_DIR, f"{prefix}_gauss_full_acc.npz")

#     # 只有当三种文件都存在时才绘图
#     if not (os.path.exists(diag_path) and os.path.exists(full_path)):
#         print(f"跳过 {prefix}（缺少 diag 或 full 文件）")
#         continue

#     # 加载原始 Acc
#     orig_data = np.load(orig_path)
#     # key 通常是 'Acc' 或 'acc'
#     key_orig = "Acc" if "Acc" in orig_data else orig_data.files[0]
#     orig_acc = orig_data[key_orig]   # shape: (n_iter, n_layers)
#     mean_orig = orig_acc.mean(axis=0)  # 每层平均，shape: (n_layers,)

#     # 加载 gauss_diag Acc
#     diag_data = np.load(diag_path)
#     # keys: 'layer0_acc', 'layer1_acc', ...
#     layers = sorted([int(k.split("layer")[1].split("_")[0]) for k in diag_data.files])
#     mean_diag = np.array([diag_data[f"layer{l}_acc"].mean() for l in layers])

#     # 加载 gauss_full Acc
#     full_data = np.load(full_path)
#     mean_full = np.array([full_data[f"layer{l}_acc"].mean() for l in layers])

#     # 绘图
#     n_layers = len(layers)
#     x = np.arange(n_layers)
#     width = 0.25

#     plt.figure(figsize=(max(8, n_layers*0.4), 5))
#     plt.bar(x - width, mean_orig,  width, label="Original",   color="C0")
#     plt.bar(x,         mean_diag,  width, label="Gauss-Diag", color="C1")
#     plt.bar(x + width, mean_full,  width, label="Gauss-Full", color="C2")

#     plt.xlabel("Layer")
#     plt.ylabel("Mean Accuracy")
#     plt.title(f"Accuracy by Layer for {prefix}")
#     plt.xticks(x, layers, rotation=45)
#     plt.ylim(0, 1)
#     plt.legend()
#     plt.tight_layout()

#     out_file = os.path.join(PLOT_DIR, f"{prefix}_layered_acc_compare.png")
#     plt.savefig(out_file)
#     plt.close()
#     print(f"已生成: {out_file}")
#!/usr/bin/env python3
# plot_layered_acc.py
# 功能：针对 lab_rs 下每组实验(prefix)，将原始 Acc、gauss_diag Acc、gauss_full Acc
# 按层绘制分组柱状图，每层展示三种方法的 mean accuracy，图像保存在 lab_rs/plot。

#!/usr/bin/env python3
# plot_layered_acc.py
# 功能：针对 lab_rs 下每组实验(prefix)，将原始 Acc、gauss_diag Acc、gauss_full Acc
# 按层绘制分组柱状图，每层展示三种方法的 mean accuracy，
# 图像保存在 ./plot/Acc 下，文件名为 {prefix}_layered_acc_compare.png 。

#!/usr/bin/env python3
# plot_layered_acc.py

#!/usr/bin/env python3
# plot_layered_acc_compare.py  —— 原始 / Gauss / Stacking 五柱对比
import os, glob, re
import numpy as np
import matplotlib.pyplot as plt

LAB_RS_DIR = "./lab_rs"
PLOT_DIR   = os.path.join("plot", "Acc")
os.makedirs(PLOT_DIR, exist_ok=True)

# 1) 找到所有原始 acc 文件（排除 *_gauss_*_results.npz 以及 *_stack_results.npz）
orig_paths = [
    p for p in glob.glob(os.path.join(LAB_RS_DIR, "*_acc.npz"))
    if not re.search(r"gauss_(diag|full|poet)", p) and "_stack_" not in p
]

def read_layer_mean(npz_path, pref_regex="layer\d+_acc(_test)?$"):
    """从 npz 中读出每层 acc 数组并求 mean"""
    data = np.load(npz_path)
    keys = sorted(
        [k for k in data.files if re.match(pref_regex, k)],
        key=lambda k: int(re.findall(r"layer(\d+)_", k)[0])
    )
    if not keys:                      # fallback: 直接用第 0 个数组
        arr = data[data.files[0]]
        return arr.mean(0) if arr.ndim == 2 else arr
    return np.array([data[k].mean() for k in keys])

for orig_path in sorted(orig_paths):
    prefix     = os.path.basename(orig_path)[:-len("_acc.npz")]
    diag_path  = os.path.join(LAB_RS_DIR, f"{prefix}_gauss_diag_results.npz")
    full_path  = os.path.join(LAB_RS_DIR, f"{prefix}_gauss_full_results.npz")
    poet_path  = os.path.join(LAB_RS_DIR, f"{prefix}_gauss_poet_results.npz")
    stack_path = os.path.join(LAB_RS_DIR, f"{prefix}_gauss_stacking_results.npz")

    # 若缺某类结果则跳过（维持原有行为）
    for tag, path in [("diag",diag_path),("full",full_path),
                      ("poet",poet_path),("stacking",stack_path)]:
        if not os.path.exists(path):
            print(f"[SKIP] {prefix}: 缺少 {tag} 结果文件")
            break
    else:  # 所有文件都在 → 继续
        # 读取
        mean_orig  = read_layer_mean(orig_path)
        mean_diag  = read_layer_mean(diag_path)
        mean_full  = read_layer_mean(full_path)
        mean_poet  = read_layer_mean(poet_path)
        # stack 先尝试读 layer{i}_acc_test, fallback 到 layer{i}_acc
        mean_stack = read_layer_mean(stack_path, "layer\\d+_acc_test$|layer\\d+_acc$")

        # 统一层数
        n_layers = len(mean_orig)
        layers   = list(range(n_layers))
        x = np.arange(n_layers)
        width = 0.15

        plt.figure(figsize=(max(8, n_layers * 0.45), 3))
        plt.bar(x - width, mean_orig,  width, label="Original")
        plt.bar(x ,   mean_diag,  width, label="GCS")
        #plt.bar(x+ width,           mean_full,  width, label="Gauss-Full")
        #plt.bar(x + width,   mean_poet,  width, label="Gauss-Poet")
        plt.bar(x + width, mean_stack, width, label="Ensemble")

        plt.xlabel("Layer")
        plt.ylabel("Mean Accuracy")
        plt.title(f"Accuracy by Layer: {prefix}")
        plt.xticks(x, layers, rotation=45)
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()

        out_file = os.path.join(PLOT_DIR, f"{prefix}_layered_acc_compare.png")
        plt.savefig(out_file, dpi=150)
        plt.close()
        print(f"[SAVED] {out_file}")


