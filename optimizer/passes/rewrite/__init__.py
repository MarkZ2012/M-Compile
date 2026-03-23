# optimizer/passes/rewrite/__init__.py
"""
Rewrite passes 自动注册。

使用方式（compile.py）：
    from my_ai_compiler.optimizer.passes import rewrite
    pm.add_pass_by_name("layout_nchw_to_nhwc")

注意：layout 类 pass 须在所有其他 pass 之前注册 / 执行。
"""
from ...pass_manager import FunctionPass, register_pass
from .layout_transform import layout_nchw_to_nhwc

register_pass("layout_nchw_to_nhwc", FunctionPass("layout_nchw_to_nhwc", layout_nchw_to_nhwc))
