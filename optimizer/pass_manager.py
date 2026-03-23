"""
Pass Manager / Pass Registry - 优化 pass 管理框架

向后兼容说明：
    原有接口 add_pass() / add_function_pass() 保持不变，已有调用代码无需修改。
    新增接口 add_pass_by_name() 支持通过注册名动态插入 pass。

Pass 推荐执行顺序（在 compile.py 中按此顺序注册）：
    layout_transform → constant_fold → dead_code_elim
        → conv_bn_relu_fusion → gemm_fusion → (ptq)

    原因：
    1. layout_transform 先统一内存布局，后续 pass 只需面对单一格式。
    2. constant_fold / dead_code_elim 缩减图规模，减少 fusion 的搜索空间。
    3. fusion 依赖 BN 参数已为常量（constant_fold 之后才成立）。
    4. 量化在最后，面对的是已经融合、精简后的图。
"""
from typing import List, Callable, Dict
from ..frontend.graph_ir import Graph


# ─────────────────────────────────────────────────────────────────────────────
# 全局 Pass 注册表
# ─────────────────────────────────────────────────────────────────────────────

_PASS_REGISTRY: Dict[str, "Pass"] = {}


def register_pass(name: str, pass_obj: "Pass") -> None:
    """
    向全局注册表注册一个 pass。

    每个 passes/ 子包的 __init__.py 在被 import 时自动调用本函数，
    compile.py 只需 import 对应子包，pass 就已注册完毕，
    之后用 add_pass_by_name(name) 即可启用。
    """
    if name in _PASS_REGISTRY:
        raise ValueError(
            f"Pass '{name}' is already registered. "
            "Use a unique name or explicitly overwrite via _PASS_REGISTRY."
        )
    _PASS_REGISTRY[name] = pass_obj


def get_pass(name: str) -> "Pass":
    """按名字取出已注册的 pass，不存在时给出可用列表提示。"""
    if name not in _PASS_REGISTRY:
        available = list(_PASS_REGISTRY.keys())
        raise KeyError(
            f"Pass '{name}' not found in registry.\n"
            f"Available passes: {available}"
        )
    return _PASS_REGISTRY[name]


def list_passes() -> List[str]:
    """返回当前已注册的所有 pass 名称（按注册顺序）。"""
    return list(_PASS_REGISTRY.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Pass 基类与函数式 Pass
# ─────────────────────────────────────────────────────────────────────────────

class Pass:
    """
    优化 pass 基类。

    Attributes:
        name     : pass 唯一标识符，与注册表 key 一致。
        priority : 保留字段，方便未来实现自动排序（数值越小越早执行）。
    """

    def __init__(self, name: str, priority: int = 0):
        self.name = name
        self.priority = priority

    def run(self, graph: Graph) -> Graph:
        """执行优化，返回（可能经过修改的）图。子类必须实现。"""
        raise NotImplementedError(f"Pass '{self.name}' did not implement run()")


class FunctionPass(Pass):
    """
    将普通函数包装为 Pass 对象，保持与原接口的兼容。

    原有代码：
        pm.add_function_pass("constant_fold", constant_fold)
    等价新写法（可选）：
        register_pass("constant_fold", FunctionPass("constant_fold", constant_fold))
        pm.add_pass_by_name("constant_fold")
    """

    def __init__(self, name: str, func: Callable[[Graph], Graph], priority: int = 0):
        super().__init__(name, priority)
        self.func = func

    def run(self, graph: Graph) -> Graph:
        return self.func(graph)


# ─────────────────────────────────────────────────────────────────────────────
# Pass Manager
# ─────────────────────────────────────────────────────────────────────────────

class PassManager:
    """
    Pass 管理器：维护一个有序的 pass 列表，依次执行。

    向后兼容接口（勿删）：
        add_pass(pass_obj)
        add_function_pass(name, func)

    新增接口（推荐用于新 pass）：
        add_pass_by_name(name)   —— 从注册表按名字动态插入
    """

    def __init__(self):
        self.passes: List[Pass] = []

    # ── 原有接口（保持签名不变）─────────────────────────────
    def add_pass(self, pass_obj: Pass) -> None:
        """直接添加一个 Pass 实例（原有接口，不变）。"""
        self.passes.append(pass_obj)

    def add_function_pass(self, name: str, func: Callable[[Graph], Graph]) -> None:
        """将函数包装为 FunctionPass 并添加（原有接口，不变）。"""
        self.passes.append(FunctionPass(name, func))

    # ── 新增接口 ────────────────────────────────────────────
    def add_pass_by_name(self, name: str) -> None:
        """
        从全局注册表按名字动态添加 pass。

        使用前需保证对应子包已被 import（import 会触发注册）：
            from my_ai_compiler.optimizer.passes import fusion
            pm.add_pass_by_name("conv_bn_relu_fusion")
        """
        self.passes.append(get_pass(name))

    def add_passes_by_name(self, names: List[str]) -> None:
        """批量按名字添加，顺序即执行顺序。"""
        for name in names:
            self.add_pass_by_name(name)

    # ── 执行 ────────────────────────────────────────────────
    def run(self, graph: Graph) -> Graph:
        """依次执行所有已注册的 pass，任意 pass 抛出异常会向上传播。"""
        total = len(self.passes)
        print(f"[PassManager] Running {total} optimization passes...")

        for i, pass_obj in enumerate(self.passes):
            print(f"  [{i + 1}/{total}] Running pass: {pass_obj.name}")
            graph = pass_obj.run(graph)

        print("[PassManager] Optimization complete.")
        return graph
