# utils/settings.py

import os
import json
import sys

# ——————————————————————————————————————
# 2. 文本输出目录（RAW_DATA_DIR）  
#    填写你希望生成的 jsonl/pkl 文件保存到哪个文件夹  
#    可以是相对路径也可以是绝对路径  
# ——————————————————————————————————————
TEXT_DIR = "/gpfs/projects/p32737/del6500_home/CD/dataset/raw"  # 例如 "data/raw" 或 "/home/you/project/data/raw"

# ——————————————————————————————————————
# 3. 概念定义文件路径（concept_gen.json）  
#    填写 preprocess 文件夹下的 concept_gen.json 的位置  
# ——————————————————————————————————————
C_FILE = "/gpfs/projects/p32737/del6500_home/CD/preprocess/concept_gen.json"  # 例如 "preprocess/concept_gen.json" 或 "./preprocess/concept_gen.json"
CONCEPTS = []
def get_all_concepts(file):
    try:
        with open(file, 'r', encoding='utf-8') as f:
            concepts = json.load(f)
            if concepts:
                c_values = [item for _, v in concepts.items() for item in v]
                return c_values
            else:
                sys.stderr.write(f"{file} is empty!")
                sys.stderr.flush()
                return
    except Exception as e:
        sys.stderr.write(f"Error reading {file}: {e}")
        sys.stderr.flush()
CONCEPTS = get_all_concepts(C_FILE)