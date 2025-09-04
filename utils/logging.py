import logging
import os
from datetime import datetime
import sys


def setup_central_logger(name, log_dir='logs', level=logging.DEBUG):
    """
    创建一个中央日志记录器，将日志同时输出到文件和控制台。
    不依赖 MPI，所有进程都以 rank=0 输出。
    """
    # 确保日志目录存在
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 使用时间戳区分不同运行
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 固定 rank 为 0
    rank = 0
    # 日志文件名，不再带 rank 信息
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")

    # 创建根 logger 并设置级别
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # 文件处理器 (INFO 及以上)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # 控制台处理器 (INFO 及以上)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # 格式化器：添加时间、rank、logger 名称和级别
    formatter = logging.Formatter('%(asctime)s - Rank %(rank)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 自定义过滤器，将 fixed rank 注入到每条记录
    class RankFilter(logging.Filter):
        def __init__(self, rank):
            super().__init__()
            self.rank = rank
        def filter(self, record):
            record.rank = self.rank
            return True

    rank_filter = RankFilter(rank)
    file_handler.addFilter(rank_filter)
    console_handler.addFilter(rank_filter)

    # 将处理器添加到根 logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    return root_logger


# 全局 logger
logger = setup_central_logger('main')


def get_error_location():
    """
    获取调用者的文件名、函数名及行号，用于错误定位。
    """
    frame = sys._getframe(2)
    return f"Error in {frame.f_code.co_filename}, {frame.f_code.co_name}, line {frame.f_lineno}"


def log_error(message):
    """记录 ERROR 级别日志，并附加代码位置信息"""
    error_location = get_error_location()
    logger.error(f"{message}. {error_location}")


def log_info(message):
    """记录 INFO 级别日志"""
    logger.info(message)


def log_warning(message):
    """记录 WARNING 级别日志"""
    logger.warning(message)


def log_debug(message):
    """记录 DEBUG 级别日志"""
    logger.debug(message)
