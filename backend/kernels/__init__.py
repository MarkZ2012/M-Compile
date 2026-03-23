"""
backend/kernels/__init__.py

KernelRegistry: 按 target 名称查找对应的 KernelSpec 集合。
用法：
    from my_ai_compiler.backend.kernels import get_kernel_registry
    reg = get_kernel_registry("generic")
    spec = reg.get("conv")           # -> KernelSpec
"""
from .base import KernelSpec, KernelRegistry
from .generic import registry as _generic_registry

_REGISTRIES = {
    "generic": _generic_registry,
}

def get_kernel_registry(target_name: str) -> KernelRegistry:
    """
    返回指定 target 的 KernelRegistry。
    未找到时抛出 KeyError，方便在 Target 初始化时快速报错。
    """
    try:
        return _REGISTRIES[target_name]
    except KeyError:
        raise KeyError(
            f"[KernelRegistry] Unknown target '{target_name}'. "
            f"Available: {list(_REGISTRIES.keys())}"
        )

def register_target(target_name: str, registry: KernelRegistry) -> None:
    """供外部（如 x86_avx / arm_neon 模块）注册自己的 registry。"""
    _REGISTRIES[target_name] = registry
