#!/usr/bin/env python
# coding: utf-8
# -----------------------------------------------
# 详细注释版本：每行代码都已添加中文解释
# -----------------------------------------------
import os  # 导入 os 模块，用于文件和路径操作
import sys,pprint  # 导入 sys 模块，用于操作 Python 运行环境
pprint.pprint(sys.path[:5])
# 获取项目根目录的绝对路径，并将其添加到 sys.path 中，方便导入项目内部模块
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # 计算当前脚本上一级目录的绝对路径
#sys.path.insert(0,project_root)  # 将项目根目录插入到模块搜索路径最前面
sys.path.append(project_root) 
import re  # 导入正则表达式模块，用于文本匹配与提取
import openai  # 导入 OpenAI SDK，用于调用 OpenAI 接口
  # 导入 dotenv，用于加载环境变量文件
from utils.files import read_pkl, write_pkl, read_json, write_json  # 从 utils.files 中导入文件读写函数
# 加载 .env 后，若你想直接写在脚本里，可用：
#OPENAI_KEY = "yourkey"
from utils.settings import TEXT_DIR, C_FILE  # 从 utils.settings 中导入常量配置
import pickle  # 导入 pickle，用于序列化和反序列化 Python 对象
import argparse  # 导入 argparse，用于解析命令行参数
import threading  # 导入 threading 模块，用于多线程支持
from queue import Queue  # 从 queue 模块中导入 Queue，用于线程间任务队列
from utils.logging import log_info, log_error  # 导入日志记录函数，用于输出信息和错误

# 加载 .env 文件中的环境变量，例如 OPENAI_API_KEY


# 全局变量：指定每个概念正负样本的总需求数量
# total = SAMPLE_SIZE  # 原作者注释：总样本量
total = 5000  # 实际使用的样本数上限

# 定义读取概念列表的函数，支持 json 或 pickle 格式
def read_concepts(file_path):
    """
    读取概念文件，支持 JSON 和 PKL 格式
    :param file_path: 概念文件路径
    :return: 解析后的 Python 对象（dict 或 list）
    """
    if file_path.endswith(".json"):
        return read_json(file_path)  # 使用自定义 JSON 读取函数
    if file_path.endswith(".pkl"):
        return read_pkl(file_path)  # 使用自定义 pickle 读取函数

# 定义处理 GPT 返回文本并提取带编号的句子的函数
def extract(text_str):
    """
    从 GPT 返回的多行字符串中提取以数字加点开头的句子
    :param text_str: GPT 返回的文本
    :return: 提取后的句子列表
    """
    lines = text_str.split('\n')  # 按行拆分
    pattern = r'^\d+\.\s*'  # 匹配以数字和点开头的模式
    results = []  # 存放提取的句子
    for line in lines:
        if re.search(pattern, line):  # 如果行匹配编号模式
            cleaned = re.sub(pattern, '', line)  # 删除编号前缀
            results.append(cleaned)  # 添加到结果列表
    return results  # 返回提取列表

# 定义读取已有数据并计算剩余样本数的函数
def read_data(file_path: str, total: int, concept: str):
    """
    如果文件存在，则加载已有数据并计算正负样本还需要生成的数量
    :param file_path: pickle 文件路径
    :param total: 每个类别所需总样本数
    :param concept: 当前概念名称
    :return: (已加载数据, 剩余正样本数, 剩余负样本数)
    """
    # 初始化结果结构，包含正、负样本列表
    data = { concept: {"positive": [], "negative": []} }
    # 如果文件不存在，直接返回初始结构和 total 数量
    if not os.path.exists(file_path):
        return data, total, total
    else:
        pos_count, neg_count = 0, 0  # 初始化已生成数量
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)  # 反序列化已有数据
                if concept in data:
                    pos_list = data[concept].get("positive", [])  # 获取正样本列表
                    neg_list = data[concept].get("negative", [])  # 获取负样本列表
                    # 如果是列表，统计长度
                    if isinstance(pos_list, list):
                        pos_count = len(pos_list)
                    if isinstance(neg_list, list):
                        neg_count = len(neg_list)
                log_info(f'Concept {concept}, Pos: {pos_count}, Neg: {neg_count}')
        except EOFError:
            log_error("pickle 文件可能已损坏，将重新生成数据")
        # 返回已有数据，以及剩余需要的数量
        return data, total - pos_count, total - neg_count

# 根据正/负标识获取对应的 prompt
def get_prompt(concept, prompts, positive=True):
    """
    获取当前概念对应的正向或负向模板文本
    :param concept: 概念名称
    :param prompts: 全部 prompt 字典
    :param positive: 是否获取正向模板
    :return: 单条 prompt 文本
    """
    if positive:
        return prompts[concept]['positive']
    else:
        return prompts[concept]['negative']

# 向 OpenAI 接口发送请求并处理结果，生成正/负样本
def process_prompt(concept, num, results, remaining, prompts, positive=True):
    """
    调用 OpenAI API 生成样本，并更新结果和剩余数量
    :param concept: 概念名称
    :param num: 本次请求想生成的样本数
    :param results: 当前已生成数据结构
    :param remaining: 本方向还需生成的样本数
    :param prompts: prompt 配置字典
    :param positive: 是否为正样本
    :return: (更新后的 results, 更新后的 remaining)
    """
    prompt_text = get_prompt(concept, prompts, positive)  # 获取 prompt
    client = openai.OpenAI(api_key=OPENAI_KEY)  # 创建 OpenAI 客户端实例

    # 如果还需要生成样本
    if remaining > 0:
        # 最小请求数量设置为 5
        if remaining < 5:
            num = 5
        else:
            num = num
        # 发送 ChatCompletion 请求
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt_text}],
            # model="gpt-4-1106-preview",  # 可选模型
            model="gpt-4o",  # 使用 gpt-4o 模型
        )
        content = response.choices[0].message.content  # 提取返回文本
        extracted = extract(content)  # 提取编号句子
        # 如果提取数量 >=5，则加入结果列表
        if len(extracted) >= 5:
            key = "positive" if positive else "negative"
            results[concept][key].extend(extracted)  # 合并到对应列表
            remaining -= len(extracted)  # 更新剩余数量
    return results, remaining  # 返回更新结果

# 生成单个概念的样本数据
def generate_sample(idx, concept):
    """
    生成一个概念的正负样本，直到满足样本量要求
    :param idx: 概念索引（未使用，仅占位）
    :param concept: 概念名称
    """
    safe_name = concept.replace(' ', '-')          # ← 新增：把空格换成 -
    file_path = f'{TEXT_DIR}/{safe_name}.pkl'
    # 读取已有数据和剩余样本量
    data, rem_pos, rem_neg = read_data(file_path, total, concept)
    # 如果正负都满足，则完成
    if rem_pos <= 0 and rem_neg <= 0:
        log_info(f'Concept {concept} 已完成生成')
        return

    prompts = read_json('/gpfs/projects/p32737/del6500_home/CD/preprocess/prompt.json')  # 读取 prompt 配置文件

    while rem_pos > 0 or rem_neg > 0:
        # 日志输出当前剩余量
        log_info(f'概念 {concept} 剩余 正样本 {rem_pos}，负样本 {rem_neg}')
        # 本次批量请求数量，最多 100
        num_p = min(rem_pos, 100)
        num_n = min(rem_neg, 100)

        # 生成正样本和负样本
        data, rem_pos = process_prompt(concept, num_p, data, rem_pos, prompts, positive=True)
        data, rem_neg = process_prompt(concept, num_n, data, rem_neg, prompts, positive=False)

        write_pkl(data, file_path)  # 保存到本地 pickle

    # 完成后日志
    log_info(f'Concept {concept} 生成完成')

# 工作线程函数，用于多线程消费队列任务
def worker(task_queue):
    """
    持续从队列获取任务并执行，直到遇到 None
    :param task_queue: Queue 实例
    """
    while True:
        task = task_queue.get()  # 获取一个任务
        if task is None:
            break  # None 作为结束信号
        idx, concept = task  # 解包任务
        try:
            generate_sample(idx, concept)  # 调用生成函数
        finally:
            task_queue.task_done()  # 通知任务完成

# 多线程并发生成上下文数据
def generate_contexts_threaded(concepts, num_threads):
    """
    使用线程池并发处理多个概念
    :param concepts: 概念列表或字典
    :param num_threads: 并发线程数
    """
    queue = Queue()  # 创建任务队列

    # 将概念加入队列，支持多种结构
    for key, val in concepts.items():
        if isinstance(val, list):
            for item in val:
                queue.put((key, item))
        else:
            queue.put((key, val))

    # 启动线程
    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=worker, args=(queue,))
        t.daemon = True  # 设置为守护线程
        t.start()
        threads.append(t)

    # 添加结束信号
    for _ in range(num_threads):
        queue.put(None)

    # 等待所有任务完成
    queue.join()
    log_info("所有任务完成。")

# 主函数，解析命令行参数并执行入口逻辑
def main(args):
    # 根据参数决定从文件读取概念或使用默认列表
    if args.file:
        concepts = read_concepts(C_FILE)  # 从配置文件读取
    else:
        # 默认概念示例，写入 C_FILE 以备下次
        concepts = { ... }
        write_json(concepts, C_FILE)

    # 示例覆盖概念，仅为 demo
    #concepts = {'': ['Joyful tweets']}

    num_threads = int(args.nthreads) if args.nthreads else 20  # 获取线程数
    generate_contexts_threaded(concepts, num_threads)  # 启动并发生成

# 脚本入口：解析参数并调用 main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate NLI samples')  # 创建解析器
    parser.add_argument('--file', type=bool, default=True, help='Path to the file containing concepts')  # 是否从文件读取
    parser.add_argument('--nthreads', type=str, help='Number of thread, default 10')  # 线程数
    args = parser.parse_args()  # 解析命令行参数
    main(args)  # 调用主函数
