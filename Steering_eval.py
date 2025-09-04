# -*- coding: utf-8 -*-
"""
功能：读取 steering_generate.py 产出的 CSV，用 GPT-4o 进行风格与连贯性打分
输入：--in_csv  (含列 method,alpha,prompt,original_output,steered_output)
输出：--out_csv (新增列 joyfulness_score,coherence_score,evaluation_comment)
环境：需要 OPENAI_API_KEY
"""

import os
import re
import csv
import time
import argparse

# 兼容新版 openai SDK (OpenAI()) 与旧版 ChatCompletion 两种写法
USE_NEW_SDK = True
try:
    from openai import OpenAI
    client = OpenAI()
except Exception:
    import openai
    USE_NEW_SDK = False

def ask_gpt_4o(prompt):
    if USE_NEW_SDK:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return resp.choices[0].message.content.strip()
    else:
        openai.api_key = os.getenv("OPENAI_API_KEY")
        resp = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return resp.choices[0].message.content.strip()

def parse_scores(text):
    joy = coh = ""
    m = re.search(r"Joyfulness\s*[:=]\s*(\d)", text, re.IGNORECASE)
    if m: joy = m.group(1)
    m = re.search(r"Coherence\s*[:=]\s*(\d)", text, re.IGNORECASE)
    if m: coh = m.group(1)
    return joy, coh

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_csv",  default="outputs/steering_results.csv")
    parser.add_argument("--out_csv", default="outputs/steering_evaluation.csv")
    parser.add_argument("--concept_desc", default="joyful", help="目标概念描述词，示例：joyful")
    parser.add_argument("--sleep", type=float, default=0.0, help="两次请求之间的休眠秒数（避免限速）")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)

    with open(args.in_csv, "r", encoding="utf-8") as fin, \
         open(args.out_csv, "w", newline="", encoding="utf-8") as fout:

        rd = csv.DictReader(fin)
        wr = csv.writer(fout)
        wr.writerow(["method","alpha","prompt","joyfulness_score","coherence_score","evaluation_comment"])

        for row in rd:
            method = row["method"]
            alpha  = row["alpha"]
            prompt = row["prompt"]
            orig   = row["original_output"]
            steer  = row["steered_output"]

            eval_prompt = (
                f"You are an impartial rater. A steered response should be more {args.concept_desc} than the original.\n"
                f"Prompt: {prompt}\n"
                f"Original response: {orig}\n"
                f"Steered response: {steer}\n\n"
                f"Rate the steered response for (1) Joyfulness and (2) Coherence, both from 1 to 5.\n"
                f"Reply strictly in the format: Joyfulness=X, Coherence=Y, Comment=... (short)."
            )

            try:
                feedback = ask_gpt_4o(eval_prompt)
            except Exception as e:
                feedback = f"Error: {e}"

            joy, coh = parse_scores(feedback)
            wr.writerow([method, alpha, prompt, joy, coh, feedback])
            print(f"[Eval] {method} α={alpha} -> Joyfulness={joy}, Coherence={coh}")
            if args.sleep > 0:
                time.sleep(args.sleep)

if __name__ == "__main__":
    main()
