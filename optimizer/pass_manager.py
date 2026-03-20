"""
Pass Manager - 优化pass管理框架
"""
from typing import List, Callable
from ..frontend.graph_ir import Graph


class Pass:
    """优化pass基类"""
    
    def __init__(self, name: str):
        self.name = name
    
    def run(self, graph: Graph) -> Graph:
        """执行优化pass"""
        raise NotImplementedError


class FunctionPass(Pass):
    """函数式pass"""
    
    def __init__(self, name: str, func: Callable[[Graph], Graph]):
        super().__init__(name)
        self.func = func
    
    def run(self, graph: Graph) -> Graph:
        return self.func(graph)


class PassManager:
    """Pass管理器"""
    
    def __init__(self):
        self.passes: List[Pass] = []
    
    def add_pass(self, pass_obj: Pass):
        """添加优化pass"""
        self.passes.append(pass_obj)
    
    def add_function_pass(self, name: str, func: Callable[[Graph], Graph]):
        """添加函数式pass"""
        self.passes.append(FunctionPass(name, func))
    
    def run(self, graph: Graph) -> Graph:
        """依次执行所有pass"""
        print(f"[PassManager] Running {len(self.passes)} optimization passes...")
        
        for i, pass_obj in enumerate(self.passes):
            print(f"  [{i+1}/{len(self.passes)}] Running pass: {pass_obj.name}")
            graph = pass_obj.run(graph)
        
        print(f"[PassManager] Optimization complete.")
        return graph