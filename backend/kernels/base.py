"""
backend/kernels/base.py

KernelSpec  : 描述一个算子的 C 运行时接口（函数签名模板、头文件、cmake glob）。
KernelRegistry : 一个 target 下所有算子 spec 的集合。

设计原则
--------
* 只存"元信息"，不生成代码 —— 代码生成仍在 Target.emit_xxx() 中完成。
* 函数签名用 Python format-string 模板，占位符由 emit_xxx 填充。
* 一个 target 可以对部分算子 override，其余 fallback 到 generic。
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class KernelSpec:
    """
    单个算子的 C 运行时描述。

    Attributes
    ----------
    op_name     : 算子标准名称，小写，如 "conv", "relu", "gemm"
    c_func      : C 函数名，如 "op_conv2d"
    signature   : C 调用模板（Python format-string）。
                  可用占位符视具体算子而定，在 emit_xxx 中 .format(**kwargs) 填充。
                  例："{inp}, {N},{C},{H},{W}, {weight}, {Co},{kH},{kW}, {bias}, {sH},{sW}, {pH},{pW}, {out}"
    headers     : 需要 #include 的头文件列表（相对于输出目录）
    cmake_glob  : CMakeLists.txt 中 file(GLOB …) 使用的 glob 模式列表
    notes       : 可选备注，供调试用
    """
    op_name:    str
    c_func:     str
    signature:  str
    headers:    List[str]       = field(default_factory=list)
    cmake_glob: List[str]       = field(default_factory=list)
    notes:      Optional[str]   = None

    def render_call(self, indent: int = 4, **kwargs) -> str:
        """
        将 signature 模板填充后，返回带缩进的完整 C 函数调用语句。

        示例
        ----
        spec.render_call(inp="buf0", N=1, C=3, H=224, W=224,
                         weight="w_conv1", Co=64, kH=7, kW=7,
                         bias="b_conv1", sH=2, sW=2, pH=3, pW=3,
                         out="buf1")
        ->  "    op_conv2d(buf0, 1,3,224,224, w_conv1, 64,7,7, b_conv1, 2,2, 3,3, buf1);"
        """
        args = self.signature.format(**kwargs)
        prefix = " " * indent
        return f"{prefix}{self.c_func}({args});"


class KernelRegistry:
    """
    一个 target 内所有算子 KernelSpec 的查找表。

    用法
    ----
    reg = KernelRegistry()
    reg.register(KernelSpec(op_name="conv", c_func="op_conv2d", ...))
    spec = reg.get("conv")      # 找不到返回 None
    spec = reg.require("conv")  # 找不到抛 KeyError
    """

    def __init__(self, target_name: str):
        self.target_name = target_name
        self._specs: Dict[str, KernelSpec] = {}

    def register(self, spec: KernelSpec) -> None:
        self._specs[spec.op_name] = spec

    def get(self, op_name: str) -> Optional[KernelSpec]:
        return self._specs.get(op_name)

    def require(self, op_name: str) -> KernelSpec:
        spec = self._specs.get(op_name)
        if spec is None:
            raise KeyError(
                f"[KernelRegistry/{self.target_name}] "
                f"No KernelSpec registered for op '{op_name}'. "
                f"Available: {list(self._specs.keys())}"
            )
        return spec

    def all_headers(self) -> List[str]:
        """聚合所有算子的头文件（去重，保序）。"""
        seen, result = set(), []
        for spec in self._specs.values():
            for h in spec.headers:
                if h not in seen:
                    seen.add(h)
                    result.append(h)
        return result

    def all_cmake_globs(self) -> List[str]:
        """聚合所有算子的 cmake glob（去重，保序）。"""
        seen, result = set(), []
        for spec in self._specs.values():
            for g in spec.cmake_glob:
                if g not in seen:
                    seen.add(g)
                    result.append(g)
        return result

    def __repr__(self) -> str:
        return (
            f"KernelRegistry(target={self.target_name!r}, "
            f"ops={list(self._specs.keys())})"
        )
