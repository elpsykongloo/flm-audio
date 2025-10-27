# Copyright (c) FLM Team, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# 该模块聚合了客户端与服务端都会用到的简单日志工具函数，
# 主要负责给终端打印的字符串增加颜色，便于在命令行中快速区分提示类型。


def colorize(text, color):
    """将传入的文本包上一段 ANSI 颜色控制码，返回彩色字符串。"""
    code = f"\033[{color}m"  # 构造颜色起始控制码，例如 "1;34" 代表加粗蓝色
    restore = "\033[0m"      # 末尾追加还原控制码，避免影响后续输出
    return "".join([code, text, restore])


def make_log(level: str, msg: str) -> str:
    """按照日志级别拼接带颜色的前缀，并返回完整日志行字符串。"""
    if level == "warning":
        prefix = colorize("[Warn]", "1;31")  # 警告统一使用红色醒目显示
    elif level == "info":
        prefix = colorize("[Info]", "1;34")  # 普通信息使用蓝色
    elif level == "error":
        prefix = colorize("[Err ]", "1;31")  # 错误同样使用红色，但前缀不同以示区分
    else:
        raise ValueError(f"Unknown level {level}")
    return prefix + " " + msg


def log(level: str, msg: str) -> None:
    """直接打印一条日志，内部调用 ``make_log`` 统一处理格式。"""
    print(make_log(level, msg))
