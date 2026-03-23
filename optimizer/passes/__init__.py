# optimizer/passes/__init__.py
"""
内置 pass 自动注册入口。

只要上层代码执行了：
    from my_ai_compiler.optimizer import passes
或者：
    from my_ai_compiler.optimizer.passes import constant_fold

本文件就会被执行，两个内置 pass 即自动注册到全局注册表，
无需在 compile.py 里手动 register_pass()。
"""
from ..pass_manager import FunctionPass, register_pass
from .constant_fold import constant_fold
from .dead_code_elim import dead_code_elim

register_pass("constant_fold",  FunctionPass("constant_fold",  constant_fold))
register_pass("dead_code_elim", FunctionPass("dead_code_elim", dead_code_elim))
