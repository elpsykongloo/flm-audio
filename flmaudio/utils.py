# Copyright (c) FLM Team, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


def colorize(text, color):
    code = f"\033[{color}m"
    restore = "\033[0m"
    return "".join([code, text, restore])


def make_log(level: str, msg: str) -> str:
    if level == "warning":
        prefix = colorize("[Warn]", "1;31")
    elif level == "info":
        prefix = colorize("[Info]", "1;34")
    elif level == "error":
        prefix = colorize("[Err ]", "1;31")
    else:
        raise ValueError(f"Unknown level {level}")
    return prefix + " " + msg


def log(level: str, msg: str) -> None:
    """Log something with a given level."""
    print(make_log(level, msg))
