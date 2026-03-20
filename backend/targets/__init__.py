"""
平台注册表 - 将平台名称字符串映射到对应的 Target 实例。

用法::

    from my_ai_compiler.backend.targets import get_target

    target = get_target("generic")   # -> GenericCTarget()
    target = get_target("x86_avx")   # -> X86AvxTarget()
    target = get_target("arm_neon")  # -> ArmNeonTarget()
"""
from .base import BaseTarget
from .generic_c import GenericCTarget
from .x86_avx import X86AvxTarget
from .arm_neon import ArmNeonTarget

# ------------------------------------------------------------------
# 注册表：平台名称 -> Target 类
# ------------------------------------------------------------------
_REGISTRY = {
    "generic":   GenericCTarget,
    "generic_c": GenericCTarget,
    "x86_avx":   X86AvxTarget,
    "x86":       X86AvxTarget,
    "arm_neon":  ArmNeonTarget,
    "arm":       ArmNeonTarget,
}


def get_target(name: str) -> BaseTarget:
    """
    根据平台名称返回已实例化的 Target 对象。

    Args:
        name: 平台名称，支持 'generic' / 'generic_c' / 'x86_avx' / 'x86' /
              'arm_neon' / 'arm'

    Returns:
        对应平台的 BaseTarget 子类实例

    Raises:
        ValueError: 若 name 未注册
    """
    key = name.lower().strip()
    cls = _REGISTRY.get(key)
    if cls is None:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(
            f"Unknown target '{name}'. Available targets: {available}"
        )
    return cls()


def list_targets():
    """返回所有已注册的平台名称列表。"""
    return sorted(set(_REGISTRY.keys()))


__all__ = ["BaseTarget", "GenericCTarget", "X86AvxTarget", "ArmNeonTarget",
           "get_target", "list_targets"]
