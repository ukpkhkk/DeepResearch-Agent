#!/usr/bin/env python
#***********************************************
#      Filename: logging.py
#   Description: 日志打印工具
#***********************************************


"""
Deep Research 的集中式日志配置。
默认情况下提供 stdout/stderr 控制台处理程序，也可以重定向到文件。
日志打印等级可通过 `DEEP_RESEARCH_LOG_LEVEL` 环境变量来控制。
"""

from __future__ import annotations

import importlib.util
import os
import sys
import sysconfig
from pathlib import Path
from typing import Optional


def _load_stdlib_logging():
    """加载标准库日志"""
    stdlib_logging_path = Path(sysconfig.get_paths()["stdlib"]) / "logging" / "__init__.py"
    spec = importlib.util.spec_from_file_location("_stdlib_logging", stdlib_logging_path)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load stdlib logging module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


if __name__ == "logging":
    std_logging = _load_stdlib_logging()
    sys.modules["logging"] = std_logging
    globals().update(std_logging.__dict__)

else:
    _logging = _load_stdlib_logging()

    # 导出常用名称，以便在导入 deep_research.logging 时方便使用 
    getLogger = _logging.getLogger
    Logger = _logging.Logger
    StreamHandler = _logging.StreamHandler
    FileHandler = _logging.FileHandler
    Formatter = _logging.Formatter

    _LOG_CONFIGURED = False
    _LOG_DIR = Path("/log/deepresearch")
    _DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    _DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"
    _ENV_LEVEL_KEY = "DEEP_RESEARCH_LOG_LEVEL"


    class _MaxLevelFilter(_logging.Filter):
        """只允许不大于设定级别的日志打印"""

        def __init__(self, level: int) -> None:
            super().__init__()
            self.level = level

        def filter(self, record: _logging.LogRecord) -> bool:  # noqa: A003  (filter name)
            return record.levelno <= self.level


    def _parse_level(level: Optional[str | int]) -> int:
        """解析日志的等级"""

        env_level = level if level is not None else os.getenv(_ENV_LEVEL_KEY, "INFO")

        if isinstance(env_level, int):
            return env_level

        level_name = str(env_level).upper()
        resolved = _logging._nameToLevel.get(level_name)  # type: ignore[attr-defined]
        if resolved is None:
            return _logging.INFO
        if resolved == 0 and level_name not in _logging._nameToLevel:  # type: ignore[attr-defined]
            return _logging.INFO
        return resolved


    def setup_logging(level: Optional[str | int] = None) -> _logging.Logger:
        """
        配置root logger，包括console和文件
        - Console：INFO 及以下级别输出到标准输出 (stdout)；WARNING 及以上级别输出到标准错误输出 (stderr)。
        - File logging：仅当 ``/log/deepresearch/`` 存在时才启用。
        """

        global _LOG_CONFIGURED
        if _LOG_CONFIGURED:
            return _logging.getLogger()

        root = _logging.getLogger()
        root.setLevel(_parse_level(level))

        # 设置打印格式
        formatter = _logging.Formatter(fmt=_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT)

        # INFO及以下写到stdout
        stdout_handler = _logging.StreamHandler(stream=sys.stdout)
        stdout_handler.setLevel(_logging.DEBUG)
        stdout_handler.addFilter(_MaxLevelFilter(_logging.INFO))
        stdout_handler.setFormatter(formatter)
        root.addHandler(stdout_handler)

        # WARNING及以上写到stderr 
        stderr_handler = _logging.StreamHandler(stream=sys.stderr)
        stderr_handler.setLevel(_logging.WARNING)
        stderr_handler.setFormatter(formatter)
        root.addHandler(stderr_handler)

        # 写到文件 
        if _LOG_DIR.exists() and _LOG_DIR.is_dir():
            try:
                app_log_path = _LOG_DIR / "app.log"
                error_log_path = _LOG_DIR / "app.error.log"

                file_handler = _logging.FileHandler(app_log_path)
                file_handler.setLevel(_logging.DEBUG)
                file_handler.setFormatter(formatter)
                root.addHandler(file_handler)

                error_file_handler = _logging.FileHandler(error_log_path)
                error_file_handler.setLevel(_logging.WARNING)
                error_file_handler.setFormatter(formatter)
                root.addHandler(error_file_handler)
            except OSError:
                # 如果无法写入文件，则继续仅使用Console
                pass

        _LOG_CONFIGURED = True
        return root


    def get_logger(name: Optional[str] = None) -> _logging.Logger:
        """返回一个具有全局配置的logger."""

        setup_logging()
        return _logging.getLogger(name)

    __all__ = [
        "getLogger",
        "Logger",
        "StreamHandler",
        "FileHandler",
        "Formatter",
        "setup_logging",
        "get_logger",
    ]
