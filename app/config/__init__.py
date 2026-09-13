import sys
from pathlib import Path

from app.config import config
from app.utils import utils
from app.utils.logging_utils import configure_terminal_logger, format_log_record
from loguru import logger


def __init_logger():
    _log_file = utils.storage_dir("logs/server.log")
    _lvl = config.log_level

    configure_terminal_logger(
        sys.stdout,
        level=_lvl,
        colorize=True,
    )

    # docker 的 json-file 捕获在高并发突发写入下会丢行（2026-09-10 实测丢
    # 失约 60-70% 的 rerank/搜索日志），文件 sink 落在 bind-mounted storage，
    # 保证任务日志可事后完整取证；rotation/retention/enqueue 复用既有设计。
    Path(_log_file).parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        _log_file,
        level=_lvl,
        format=format_log_record,
        colorize=False,
        rotation="00:00",
        retention="3 days",
        backtrace=True,
        diagnose=True,
        enqueue=True,
    )


__init_logger()
