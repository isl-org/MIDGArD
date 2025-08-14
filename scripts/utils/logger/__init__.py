"""
Taken from https://github.com/JiahuiLei/NAP/tree/main/logger
"""

from .logger import Logger


def get_logger(config):
    return Logger(config)
