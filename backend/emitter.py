"""
emitter.py - 代码生成主驱动

调用流程：
    compile.py
        └─> emit_c_code(graph, output_dir, model_name, target)
                └─> CEmitter(output_dir, target).emit(graph, model_name)
                        ├─ core/allocator.py  (BufferAllocator)
                        ├─ core/shape_infer.py (ShapeInferer，被 allocator 内部调用)
                        └─ targets/generic_c.py (或其他平台 Target)

此文件只负责「控制流程」：
  1. 创建 BufferAllocator
  2. 遍历 Graph 节点，将每个节点分发给 target.emit_xxx(node, alloc)
  3. 将生成的 C 代码、头文件、权重二进制、CMakeLists.txt 写入磁盘
"""
import os
import numpy as np
from typing import Optional

from ..frontend.graph_ir import Graph
from .core.allocator import BufferAllocator, _safe, _prod
from .targets.base import BaseTarget


# ---------------------------------------------------------------------------
# 主驱动类
# ---------------------------------------------------------------------------

class CEmitter:
    """
    将 Graph IR 翻译为 C 源码，并把权重写成二进制文件。

    Args:
        output_dir : 输出目录（自动创建）
        target     : BaseTarget 实例；为 None 时使用 GenericCTarget
    """

    def __init__(self, output_dir: str, target: Optional[BaseTarget] = None):
        self.output_dir = output_dir
        if target is None:
            from .targets.generic_c import GenericCTarget
            self.target: BaseTarget = GenericCTarget()
        else:
            self.target = target

    # ------------------------------------------------------------------
    # 公开入口
    # ------------------------------------------------------------------

    def emit(self, graph: Graph, model_name: str = "model") -> None:
        """
        生成完整的运行时 C 工程：
          {output_dir}/{model_name}.h
          {output_dir}/{model_name}.c
          {output_dir}/weights/*.bin
          {output_dir}/CMakeLists.txt
        """
        os.makedirs(self.output_dir, exist_ok=True)
        self._emit_header(graph, model_name)
        self._emit_implementation(graph, model_name)
        self._save_weights(graph)
        self._emit_cmake(model_name)
        print(
            f"[CEmitter/{self.target.target_name}] "
            f"Generated C code in '{self.output_dir}'"
        )

    # ------------------------------------------------------------------
    # 头文件
    # ------------------------------------------------------------------

    def _emit_header(self, graph: Graph, model_name: str) -> None:
        guard = model_name.upper()
        h = (
            f"#ifndef {guard}_H\n"
            f"#define {guard}_H\n"
            f"#include <stdint.h>\n"
            f"#include <stdlib.h>\n"
            f"typedef struct {{ float* data; int* shape; int ndim; }} Tensor;\n"
            f"int  {model_name}_init(void);\n"
            f"int  {model_name}_run(Tensor* input, Tensor* output);\n"
            f"void {model_name}_cleanup(void);\n"
            f"void {model_name}_forward(const float* input, float* output);\n"
            f"#endif\n"
        )
        path = os.path.join(self.output_dir, f"{model_name}.h")
        with open(path, "w", encoding="utf-8") as f:
            f.write(h)

    # ------------------------------------------------------------------
    # .c 实现文件
    # ------------------------------------------------------------------

    def _emit_implementation(self, graph: Graph, model_name: str) -> None:
        alloc = BufferAllocator(graph)
        lines = []

        # --- includes ---
        lines.append(f'#include "{model_name}.h"')
        lines.extend(self.target.get_includes())
        lines += ["#include <stdio.h>", "#include <string.h>", "#include <math.h>", ""]

        # --- 权重静态数组声明 ---
        lines.append("// Weights")
        for name, tensor in graph.tensors.items():
            if tensor.data is not None:
                lines.append(
                    f"static float {_safe(name)}[{_prod(tensor.shape)}];"
                )
        lines.append("")

        # --- 中间激活 buffer 声明 ---
        lines.append("// Intermediate buffers")
        for varname, size, shape in alloc.intermediate_bufs():
            lines.append(f"static float {varname}[{max(size, 1)}];  // {shape}")
        lines.append("")

        # --- 权重加载辅助函数 ---
        lines += [
            "static int _load(const char* p, float* b, int n) {",
            '    FILE* f = fopen(p,"rb");',
            '    if(!f){fprintf(stderr,"[ERROR] %s\\n",p);return -1;}',
            "    int got=(int)fread(b,sizeof(float),n,f); fclose(f);",
            '    if(got!=n){fprintf(stderr,"[ERROR] %s want %d got %d\\n",p,n,got);return -1;}',
            "    return 0;",
            "}",
            "",
        ]

        # --- model_init: 加载权重 ---
        lines.append(f"int {model_name}_init(void) {{")
        for name, tensor in graph.tensors.items():
            if tensor.data is not None:
                safe = _safe(name)
                size = _prod(tensor.shape)
                lines.append(
                    f'    if(_load("weights/{safe}.bin",{safe},{size}))return -1;'
                )
        lines += ["    return 0;", "}", ""]

        # --- model_forward: 逐节点派发 ---
        lines.append(f"void {model_name}_forward(const float* input, float* output) {{")
        for node in graph.nodes:
            lines.extend(self._dispatch(node, alloc))
        lines += ["}", ""]

        # --- model_run / model_cleanup ---
        lines += [
            f"int {model_name}_run(Tensor* i, Tensor* o) {{",
            f"    {model_name}_forward(i->data, o->data);",
            f"    return 0;",
            f"}}",
            f"void {model_name}_cleanup(void) {{}}",
            "",
        ]

        path = os.path.join(self.output_dir, f"{model_name}.c")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    # ------------------------------------------------------------------
    # 节点派发：op_type -> target.emit_xxx
    # ------------------------------------------------------------------

    def _dispatch(self, node, alloc) -> list:
        """
        将节点的 op_type（大驼峰，如 "Conv"）转换为 target 方法名
        emit_conv，并调用之。若方法不存在则调用 emit_unknown。
        """
        method_name = f"emit_{node.op_type.lower()}"
        emit_fn = getattr(self.target, method_name, self.target.emit_unknown)
        return emit_fn(node, alloc)

    # ------------------------------------------------------------------
    # 权重二进制
    # ------------------------------------------------------------------

    def _save_weights(self, graph: Graph) -> None:
        weights_dir = os.path.join(self.output_dir, "weights")
        os.makedirs(weights_dir, exist_ok=True)
        count = 0
        for name, tensor in graph.tensors.items():
            if tensor.data is not None:
                out_path = os.path.join(weights_dir, f"{_safe(name)}.bin")
                tensor.data.astype(np.float32).tofile(out_path)
                count += 1
        print(f"  [CEmitter] Saved {count} weight files to '{weights_dir}'")

    # ------------------------------------------------------------------
    # CMakeLists.txt
    # ------------------------------------------------------------------

    def _emit_cmake(self, model_name: str) -> None:
        content = self.target.emit_cmake(model_name)
        path    = os.path.join(self.output_dir, "CMakeLists.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)


# ---------------------------------------------------------------------------
# 便捷函数（compile.py 调用此入口）
# ---------------------------------------------------------------------------

def emit_c_code(
    graph,
    output_dir: str,
    model_name: str = "model",
    target: Optional[BaseTarget] = None,
) -> None:
    """
    compile.py 的调用入口。

    Args:
        graph      : 经过优化的 Graph IR
        output_dir : 输出目录
        model_name : 生成的 C 文件前缀（默认 "model"）
        target     : BaseTarget 实例；None 时自动选用 GenericCTarget
    """
    CEmitter(output_dir, target).emit(graph, model_name)
