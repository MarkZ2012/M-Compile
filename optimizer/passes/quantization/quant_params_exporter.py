"""
量化参数导出模块

根据审核意见，这是量化流程中的关键缺口：
  ptq.py 计算 scale/zp -> quant_params_exporter.py 导出参数 -> emitter.py 生成C代码时注入常量
"""
import os
import json
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class LayerQuantParams:
    """单层的量化参数"""
    layer_name: str
    op_type: str
    weight_scale: float = 1.0
    weight_zero_point: int = 0
    weight_scales: Optional[List[float]] = None
    weight_zero_points: Optional[List[int]] = None
    input_scale: float = 1.0
    input_zero_point: int = 0
    output_scale: float = 1.0
    output_zero_point: int = 0
    per_channel: bool = False
    bit_width: int = 8


@dataclass
class QuantParamsExport:
    """整个模型的量化参数导出"""
    model_name: str
    layers: Dict[str, LayerQuantParams] = field(default_factory=dict)
    
    def add_layer_params(self, params: LayerQuantParams):
        self.layers[params.layer_name] = params
    
    def to_dict(self) -> dict:
        result = {"model_name": self.model_name, "layers": {}}
        for name, params in self.layers.items():
            result["layers"][name] = {
                "layer_name": params.layer_name,
                "op_type": params.op_type,
                "weight_scale": params.weight_scale,
                "weight_zero_point": params.weight_zero_point,
                "weight_scales": params.weight_scales,
                "weight_zero_points": params.weight_zero_points,
                "input_scale": params.input_scale,
                "input_zero_point": params.input_zero_point,
                "output_scale": params.output_scale,
                "output_zero_point": params.output_zero_point,
                "per_channel": params.per_channel,
                "bit_width": params.bit_width,
            }
        return result


class QuantParamsExporter:
    """量化参数导出器"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def export_json(self, quant_params: QuantParamsExport) -> str:
        output_path = os.path.join(self.output_dir, "quant_params.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(quant_params.to_dict(), f, indent=2)
        print(f"  [QuantParamsExporter] Exported to {output_path}")
        return output_path
    
    def export_c_header(self, quant_params: QuantParamsExport, model_name: str) -> str:
        output_path = os.path.join(self.output_dir, "quant_params.h")
        lines = [
            f"#ifndef {model_name.upper()}_QUANT_PARAMS_H",
            f"#define {model_name.upper()}_QUANT_PARAMS_H",
            "", "#include <stdint.h>", "",
        ]
        for layer_name, params in quant_params.layers.items():
            safe_name = layer_name.replace("/", "_").replace(".", "_")
            lines.append(f"/* Layer: {layer_name} */")
            if params.per_channel and params.weight_scales:
                num_ch = len(params.weight_scales)
                lines.append(f"static const float {safe_name}_scales[{num_ch}] = {{")
                lines.append(", ".join(f"{s:.10f}f" for s in params.weight_scales))
                lines.append("};")
            else:
                lines.append(f"static const float {safe_name}_scale = {params.weight_scale:.10f}f;")
                lines.append(f"static const int32_t {safe_name}_zp = {params.weight_zero_point};")
            lines.append("")
        lines.append(f"#endif")
        with open(output_path, "w") as f:
            f.write("\n".join(lines))
        print(f"  [QuantParamsExporter] Exported C header to {output_path}")
        return output_path


def create_quant_params_from_graph(graph, config) -> QuantParamsExport:
    export = QuantParamsExport(model_name="model")
    for node in graph.nodes:
        if not hasattr(node, "attrs") or node.attrs is None:
            continue
        if not node.attrs.get("quantized", False):
            continue
        params = LayerQuantParams(
            layer_name=node.name,
            op_type=node.op_type,
            weight_scale=node.attrs.get("weight_scale", 1.0),
            weight_zero_point=node.attrs.get("weight_zero_point", 0),
            per_channel=node.attrs.get("per_channel", False),
            bit_width=config.bit_width if config else 8,
        )
        export.add_layer_params(params)
    return export