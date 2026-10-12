"""``python -m bcl`` 入口。

只做一件事: 把 ``bcl.cli.main`` 的返回值当作进程退出码。之所以单独放一个文件,
是因为 ``python -m bcl`` 会执行 ``bcl/__main__.py`` 而不会执行包本身 —— 把入口
写在这里, 包内部就不会因为"运行 CLI"而被隐式执行。
"""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
