"""
性能分析工具 (Performance Profiler)

功能：
  1. 计算每层FLOPs（浮点运算次数）
  2. 估算内存带宽需求
  3. 分析算子性能瓶颈
  4. 生成性能分析报告（支持HTML和文本格式）
  5. 支持量化模型的性能评估

用法示例：
  python performance_profiler.py model.onnx -o profile.html
  python performance_profiler.py model.onnx --compare-quantized quantized_model.onnx
  python performance_profiler.py model.onnx --target x86_avx
"""
import sys
import os
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from my_ai_compiler.frontend.onnx_parser import parse_onnx
from my_ai_compiler.frontend.graph_ir import Graph, Node, Tensor
from my_ai_compiler.backend.core.shape_infer import ShapeInferer
from my_ai_compiler.backend.core.allocator import BufferAllocator


@dataclass
class LayerProfile:
    """单层性能分析结果"""
    name: str
    op_type: str
    flops: int
    memory_read: int
    memory_write: int
    params: int
    input_shape: List[int]
    output_shape: List[int]
    is_quantized: bool = False
    
    @property
    def total_memory(self) -> int:
        return self.memory_read + self.memory_write
    
    @property
    def arithmetic_intensity(self) -> float:
        if self.total_memory == 0:
            return 0.0
        return self.flops / self.total_memory


@dataclass
class ModelProfile:
    """模型整体性能分析结果"""
    total_flops: int
    total_memory_access: int
    total_params: int
    layer_profiles: List[LayerProfile]
    bottleneck_layers: List[str]
    operator_breakdown: Dict[str, int]
    
    @property
    def memory_bound_ratio(self) -> float:
        if self.total_flops == 0:
            return 0.0
        low_intensity_count = sum(
            1 for lp in self.layer_profiles 
            if lp.arithmetic_intensity < 10
        )
        return low_intensity_count / len(self.layer_profiles) if self.layer_profiles else 0


class PerformanceProfiler:
    """性能分析器"""
    
    FLOPS_PER_OP = {
        'Conv': 'conv_flops',
        'ConvBnRelu': 'conv_flops',
        'ConvBn': 'conv_flops',
        'DepthwiseConv': 'depthwise_conv_flops',
        'Gemm': 'gemm_flops',
        'MatMul': 'matmul_flops',
        'Relu': 'elementwise_flops',
        'Clip': 'elementwise_flops',
        'Sigmoid': 'elementwise_flops',
        'Tanh': 'elementwise_flops',
        'Add': 'elementwise_flops',
        'Mul': 'elementwise_flops',
        'Sub': 'elementwise_flops',
        'Div': 'elementwise_flops',
        'MaxPool': 'pool_flops',
        'AveragePool': 'pool_flops',
        'GlobalAveragePool': 'global_pool_flops',
        'BatchNormalization': 'batchnorm_flops',
        'InstanceNormalization': 'instancenorm_flops',
        'LayerNormalization': 'layernorm_flops',
        'Softmax': 'softmax_flops',
        'Resize': 'resize_flops',
        'Concat': 'concat_flops',
        'Reshape': 'reshape_flops',
        'Transpose': 'transpose_flops',
        'Flatten': 'reshape_flops',
        'ReduceMean': 'reduce_flops',
        'Pad': 'pad_flops',
    }
    
    def __init__(self, graph: Graph):
        self.graph = graph
        self.shape_inferer = ShapeInferer(graph)
        self.allocator = BufferAllocator(graph)
        self._tensor_sizes = self._compute_tensor_sizes()
    
    def _compute_tensor_sizes(self) -> Dict[str, int]:
        """计算所有张量的元素数量"""
        sizes = {}
        for name, tensor in self.graph.tensors.items():
            if tensor.shape:
                size = 1
                for s in tensor.shape:
                    size *= max(s, 1)
                sizes[name] = size
            else:
                shape = self.shape_inferer.get(name)
                if shape:
                    size = 1
                    for s in shape:
                        size *= max(s, 1)
                    sizes[name] = size
                else:
                    sizes[name] = 1
        return sizes
    
    def _prod(self, shape: List[int]) -> int:
        r = 1
        for s in shape:
            r *= max(s, 1)
        return r
    
    def profile(self) -> ModelProfile:
        """执行性能分析"""
        layer_profiles = []
        
        for node in self.graph.nodes:
            lp = self._profile_node(node)
            layer_profiles.append(lp)
        
        total_flops = sum(lp.flops for lp in layer_profiles)
        total_memory = sum(lp.total_memory for lp in layer_profiles)
        total_params = sum(lp.params for lp in layer_profiles)
        
        bottlenecks = self._identify_bottlenecks(layer_profiles)
        operator_breakdown = self._compute_operator_breakdown(layer_profiles)
        
        return ModelProfile(
            total_flops=total_flops,
            total_memory_access=total_memory,
            total_params=total_params,
            layer_profiles=layer_profiles,
            bottleneck_layers=bottlenecks,
            operator_breakdown=operator_breakdown
        )
    
    def _profile_node(self, node: Node) -> LayerProfile:
        """分析单个节点"""
        input_shapes = [self.shape_inferer.get(inp) or [] for inp in node.inputs]
        output_shapes = [self.shape_inferer.get(out) or [] for out in node.outputs]
        
        input_shape = input_shapes[0] if input_shapes else []
        output_shape = output_shapes[0] if output_shapes else []
        
        flops = self._compute_flops(node, input_shapes, output_shapes)
        memory_read, memory_write = self._compute_memory(node, input_shapes, output_shapes)
        params = self._compute_params(node)
        
        is_quantized = node.attrs.get('quantized', False)
        
        return LayerProfile(
            name=node.name,
            op_type=node.op_type,
            flops=flops,
            memory_read=memory_read,
            memory_write=memory_write,
            params=params,
            input_shape=input_shape,
            output_shape=output_shape,
            is_quantized=is_quantized
        )
    
    def _compute_flops(self, node: Node, input_shapes: List[List[int]], 
                       output_shapes: List[List[int]]) -> int:
        """计算FLOPs"""
        op = node.op_type
        attrs = node.attrs
        
        if op in ['Conv', 'ConvBnRelu', 'ConvBn']:
            return self._conv_flops(input_shapes, attrs)
        elif op == 'DepthwiseConv':
            return self._depthwise_conv_flops(input_shapes, attrs)
        elif op in ['Gemm', 'MatMul']:
            return self._gemm_flops(input_shapes, attrs)
        elif op in ['Relu', 'Clip', 'Sigmoid', 'Tanh', 'Add', 'Mul', 'Sub', 'Div']:
            return self._elementwise_flops(input_shapes)
        elif op in ['MaxPool', 'AveragePool']:
            return self._pool_flops(input_shapes, output_shapes)
        elif op == 'GlobalAveragePool':
            return self._global_pool_flops(input_shapes)
        elif op in ['BatchNormalization', 'InstanceNormalization', 'LayerNormalization']:
            return self._norm_flops(input_shapes)
        elif op == 'Softmax':
            return self._softmax_flops(input_shapes)
        elif op == 'Resize':
            return self._resize_flops(input_shapes, output_shapes)
        elif op == 'ReduceMean':
            return self._reduce_flops(input_shapes, attrs)
        else:
            return self._elementwise_flops(input_shapes)
    
    def _conv_flops(self, shapes: List[List[int]], attrs: Dict) -> int:
        """Conv FLOPs = 2 * Cout * Hout * Wout * Cin * Kh * Kw"""
        if not shapes or not shapes[0]:
            return 0
        
        input_shape = shapes[0]
        if len(input_shape) < 4:
            return 0
        
        N, Cin, Hin, Win = input_shape[:4]
        
        Cout = attrs.get('group', 1)
        kernel_shape = attrs.get('kernel_shape', [1, 1])
        strides = attrs.get('strides', [1, 1])
        pads = attrs.get('pads', [0, 0, 0, 0])
        
        Kh, Kw = kernel_shape[0], kernel_shape[1]
        
        Hout = (Hin + pads[0] + pads[2] - Kh) // strides[0] + 1
        Wout = (Hin + pads[1] + pads[3] - Kw) // strides[1] + 1
        
        group = attrs.get('group', 1)
        Cout_per_group = Cin // group
        
        for inp_name in self.graph.tensors:
            if 'weight' in inp_name.lower() and inp_name in self._tensor_sizes:
                weight_shape = self.graph.tensors[inp_name].shape
                if len(weight_shape) >= 4:
                    Cout = weight_shape[0]
                    break
        
        flops = 2 * N * Cout * Hout * Wout * (Cin // group) * Kh * Kw
        return max(flops, 0)
    
    def _depthwise_conv_flops(self, shapes: List[List[int]], attrs: Dict) -> int:
        """Depthwise Conv FLOPs"""
        if not shapes or not shapes[0]:
            return 0
        
        input_shape = shapes[0]
        if len(input_shape) < 4:
            return 0
        
        N, C, H, W = input_shape[:4]
        kernel_shape = attrs.get('kernel_shape', [1, 1])
        Kh, Kw = kernel_shape[0], kernel_shape[1]
        
        return 2 * N * C * H * W * Kh * Kw
    
    def _gemm_flops(self, shapes: List[List[int]], attrs: Dict) -> int:
        """Gemm/MatMul FLOPs = 2 * M * N * K"""
        if len(shapes) < 2:
            return 0
        
        a_shape = shapes[0]
        b_shape = shapes[1]
        
        if len(a_shape) >= 2 and len(b_shape) >= 2:
            M = a_shape[-2]
            K = a_shape[-1]
            N = b_shape[-1]
            
            trans_a = attrs.get('transA', 0)
            trans_b = attrs.get('transB', 0)
            
            if trans_a:
                M, K = K, M
            if trans_b:
                K, N = N, K
            
            return 2 * M * N * K
        
        return 0
    
    def _elementwise_flops(self, shapes: List[List[int]]) -> int:
        """Elementwise op FLOPs"""
        if not shapes:
            return 0
        return self._prod(shapes[0]) if shapes[0] else 0
    
    def _pool_flops(self, input_shapes: List[List[int]], 
                    output_shapes: List[List[int]]) -> int:
        """Pool FLOPs"""
        if output_shapes and output_shapes[0]:
            return self._prod(output_shapes[0])
        return self._elementwise_flops(input_shapes)
    
    def _global_pool_flops(self, shapes: List[List[int]]) -> int:
        """Global Pool FLOPs"""
        if not shapes or not shapes[0]:
            return 0
        return self._prod(shapes[0])
    
    def _norm_flops(self, shapes: List[List[int]]) -> int:
        """Normalization FLOPs (约 4x elementwise)"""
        if not shapes or not shapes[0]:
            return 0
        return 4 * self._prod(shapes[0])
    
    def _softmax_flops(self, shapes: List[List[int]]) -> int:
        """Softmax FLOPs (约 5x elementwise)"""
        if not shapes or not shapes[0]:
            return 0
        return 5 * self._prod(shapes[0])
    
    def _resize_flops(self, input_shapes: List[List[int]], 
                      output_shapes: List[List[int]]) -> int:
        """Resize FLOPs"""
        if output_shapes and output_shapes[0]:
            return self._prod(output_shapes[0])
        return 0
    
    def _reduce_flops(self, shapes: List[List[int]], attrs: Dict) -> int:
        """Reduce FLOPs"""
        if not shapes or not shapes[0]:
            return 0
        return self._prod(shapes[0])
    
    def _compute_memory(self, node: Node, input_shapes: List[List[int]], 
                        output_shapes: List[List[int]]) -> Tuple[int, int]:
        """计算内存访问量"""
        memory_read = 0
        memory_write = 0
        
        for i, inp in enumerate(node.inputs):
            if i < len(input_shapes) and input_shapes[i]:
                memory_read += self._prod(input_shapes[i])
            elif inp in self._tensor_sizes:
                memory_read += self._tensor_sizes[inp]
        
        for i, out in enumerate(node.outputs):
            if i < len(output_shapes) and output_shapes[i]:
                memory_write += self._prod(output_shapes[i])
            elif out in self._tensor_sizes:
                memory_write += self._tensor_sizes[out]
        
        return memory_read, memory_write
    
    def _compute_params(self, node: Node) -> int:
        """计算参数量"""
        params = 0
        for inp in node.inputs[1:]:
            if inp in self.graph.tensors:
                tensor = self.graph.tensors[inp]
                if tensor.data is not None:
                    params += self._prod(tensor.shape)
        return params
    
    def _identify_bottlenecks(self, layer_profiles: List[LayerProfile], 
                              top_k: int = 5) -> List[str]:
        """识别性能瓶颈层"""
        sorted_by_flops = sorted(layer_profiles, key=lambda x: -x.flops)
        return [lp.name for lp in sorted_by_flops[:top_k]]
    
    def _compute_operator_breakdown(self, layer_profiles: List[LayerProfile]) -> Dict[str, int]:
        """计算各算子FLOPs占比"""
        breakdown = defaultdict(int)
        for lp in layer_profiles:
            breakdown[lp.op_type] += lp.flops
        return dict(sorted(breakdown.items(), key=lambda x: -x[1]))
    
    def generate_html_report(self, profile: ModelProfile, output_path: str,
                            title: str = "Performance Profile") -> None:
        """生成HTML性能报告"""
        html = self._generate_profile_html(profile, title)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"[Profiler] Performance report saved to: {output_path}")
    
    def _generate_profile_html(self, profile: ModelProfile, title: str) -> str:
        """生成性能分析HTML"""
        
        layer_rows = []
        for i, lp in enumerate(profile.layer_profiles):
            intensity = lp.arithmetic_intensity
            intensity_class = 'intensity-low' if intensity < 10 else ('intensity-medium' if intensity < 50 else 'intensity-high')
            
            layer_rows.append(f'''
                <tr>
                    <td>{i}</td>
                    <td style="font-family: monospace; font-size: 11px;">{lp.name[:25]}</td>
                    <td><span class="op-badge">{lp.op_type}</span></td>
                    <td>{self._format_flops(lp.flops)}</td>
                    <td>{self._format_memory(lp.memory_read)}</td>
                    <td>{self._format_memory(lp.memory_write)}</td>
                    <td>{self._format_memory(lp.params)}</td>
                    <td><span class="intensity-badge {intensity_class}">{intensity:.1f}</span></td>
                    <td>{lp.flops / profile.total_flops * 100:.1f}%</td>
                </tr>
            ''')
        
        flops_chart_data = json.dumps([
            {'op': op, 'flops': flops}
            for op, flops in list(profile.operator_breakdown.items())[:10]
        ])
        
        return f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f7fa;
            color: #333;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
        .header {{ 
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .header h1 {{ font-size: 28px; margin-bottom: 10px; }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .stat-card h3 {{ color: #666; font-size: 14px; margin-bottom: 5px; }}
        .stat-card .value {{ font-size: 24px; font-weight: bold; color: #11998e; }}
        .stat-card .sub {{ font-size: 12px; color: #999; margin-top: 5px; }}
        
        .section {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .section h2 {{ 
            font-size: 18px; 
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #11998e;
        }}
        
        .bottleneck-list {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }}
        .bottleneck-item {{
            background: #fff3cd;
            border: 1px solid #ffc107;
            padding: 8px 15px;
            border-radius: 6px;
            font-size: 13px;
        }}
        
        .layer-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}
        .layer-table th {{
            background: #f8f9fa;
            padding: 12px;
            text-align: left;
            border-bottom: 2px solid #dee2e6;
            position: sticky;
            top: 0;
        }}
        .layer-table td {{
            padding: 10px 12px;
            border-bottom: 1px solid #eee;
        }}
        .layer-table tr:hover {{ background: #f8f9fa; }}
        
        .op-badge {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
            background: #11998e;
            color: white;
        }}
        
        .intensity-badge {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
        }}
        .intensity-low {{ background: #ffcdd2; color: #c62828; }}
        .intensity-medium {{ background: #fff9c4; color: #f57f17; }}
        .intensity-high {{ background: #c8e6c9; color: #2e7d32; }}
        
        .table-container {{
            max-height: 500px;
            overflow-y: auto;
        }}
        
        .chart-container {{
            height: 300px;
            position: relative;
        }}
        .bar-chart {{
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}
        .bar-row {{
            display: flex;
            align-items: center;
        }}
        .bar-label {{ width: 100px; font-size: 12px; }}
        .bar-track {{ 
            flex: 1; 
            height: 20px; 
            background: #e0e0e0; 
            border-radius: 4px;
            overflow: hidden;
        }}
        .bar-fill {{
            height: 100%;
            background: linear-gradient(90deg, #11998e, #38ef7d);
            display: flex;
            align-items: center;
            padding-left: 8px;
            color: white;
            font-size: 10px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>⚡ {title}</h1>
            <p>Performance Analysis Report</p>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <h3>Total FLOPs</h3>
                <div class="value">{self._format_flops(profile.total_flops)}</div>
                <div class="sub">{profile.total_flops:,} operations</div>
            </div>
            <div class="stat-card">
                <h3>Memory Access</h3>
                <div class="value">{self._format_memory(profile.total_memory_access)}</div>
                <div class="sub">Read + Write</div>
            </div>
            <div class="stat-card">
                <h3>Total Parameters</h3>
                <div class="value">{self._format_memory(profile.total_params)}</div>
                <div class="sub">{profile.total_params:,} params</div>
            </div>
            <div class="stat-card">
                <h3>Memory Bound Ratio</h3>
                <div class="value">{profile.memory_bound_ratio*100:.1f}%</div>
                <div class="sub">Low intensity layers</div>
            </div>
        </div>
        
        <div class="section">
            <h2>🔥 Performance Bottlenecks (Top 5 by FLOPs)</h2>
            <div class="bottleneck-list">
                {''.join(f'<div class="bottleneck-item">⚠️ {name}</div>' for name in profile.bottleneck_layers)}
            </div>
        </div>
        
        <div class="section">
            <h2>📊 FLOPs Distribution by Operator</h2>
            <div class="chart-container">
                <div class="bar-chart">
                    {self._generate_flops_bars(profile.operator_breakdown, profile.total_flops)}
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>📋 Layer-wise Performance Profile</h2>
            <div class="table-container">
                <table class="layer-table">
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>Layer Name</th>
                            <th>Op Type</th>
                            <th>FLOPs</th>
                            <th>Mem Read</th>
                            <th>Mem Write</th>
                            <th>Params</th>
                            <th>Intensity</th>
                            <th>% Total</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(layer_rows)}
                    </tbody>
                </table>
            </div>
        </div>
    </div>
</body>
</html>'''
    
    def _generate_flops_bars(self, breakdown: Dict[str, int], total: int) -> str:
        """生成FLOPs分布条形图"""
        bars = []
        for op, flops in list(breakdown.items())[:10]:
            pct = flops / total * 100 if total > 0 else 0
            bars.append(f'''
                <div class="bar-row">
                    <span class="bar-label">{op}</span>
                    <div class="bar-track">
                        <div class="bar-fill" style="width: {pct}%;">{pct:.1f}%</div>
                    </div>
                </div>
            ''')
        return '\n'.join(bars)
    
    def _format_flops(self, flops: int) -> str:
        """格式化FLOPs"""
        if flops >= 1e9:
            return f"{flops/1e9:.2f} GFLOPs"
        elif flops >= 1e6:
            return f"{flops/1e6:.2f} MFLOPs"
        elif flops >= 1e3:
            return f"{flops/1e3:.2f} KFLOPs"
        return f"{flops} FLOPs"
    
    def _format_memory(self, bytes_count: int) -> str:
        """格式化内存大小"""
        bytes_count *= 4
        if bytes_count >= 1e9:
            return f"{bytes_count/1e9:.2f} GB"
        elif bytes_count >= 1e6:
            return f"{bytes_count/1e6:.2f} MB"
        elif bytes_count >= 1e3:
            return f"{bytes_count/1e3:.2f} KB"
        return f"{bytes_count} B"
    
    def print_summary(self, profile: ModelProfile) -> None:
        """打印性能摘要"""
        print("\n" + "="*60)
        print("⚡ Performance Profile Summary")
        print("="*60)
        print(f"Total FLOPs: {self._format_flops(profile.total_flops)}")
        print(f"Total Memory Access: {self._format_memory(profile.total_memory_access)}")
        print(f"Total Parameters: {self._format_memory(profile.total_params)}")
        print(f"Memory Bound Ratio: {profile.memory_bound_ratio*100:.1f}%")
        
        print("\n🔥 Top Bottleneck Layers:")
        for i, name in enumerate(profile.bottleneck_layers, 1):
            print(f"  {i}. {name}")
        
        print("\n📊 FLOPs Distribution:")
        for op, flops in list(profile.operator_breakdown.items())[:5]:
            pct = flops / profile.total_flops * 100 if profile.total_flops > 0 else 0
            print(f"  {op:20s}: {self._format_flops(flops):>15s} ({pct:.1f}%)")
        
        print("\n" + "="*60)


def profile_model(model_path: str, output_path: Optional[str] = None,
                  title: str = "Performance Profile", summary_only: bool = False) -> ModelProfile:
    """分析模型性能"""
    print(f"\n[Profiler] Loading model: {model_path}")
    graph = parse_onnx(model_path)
    
    profiler = PerformanceProfiler(graph)
    profile = profiler.profile()
    
    if summary_only:
        profiler.print_summary(profile)
    else:
        if output_path is None:
            output_path = model_path.replace('.onnx', '_profile.html')
        profiler.generate_html_report(profile, output_path, title)
        profiler.print_summary(profile)
    
    return profile


def compare_models(fp32_path: str, quant_path: str) -> None:
    """比较量化前后的性能差异"""
    print("\n" + "="*60)
    print("📊 Comparing FP32 vs Quantized Model")
    print("="*60)
    
    fp32_profile = profile_model(fp32_path, summary_only=True)
    quant_profile = profile_model(quant_path, summary_only=True)
    
    print("\n📈 Comparison Results:")
    print("-" * 40)
    
    flops_reduction = (1 - quant_profile.total_flops / fp32_profile.total_flops) * 100
    memory_reduction = (1 - quant_profile.total_memory_access / fp32_profile.total_memory_access) * 100
    
    print(f"FLOPs: {fp32_profile.total_flops:,} → {quant_profile.total_flops:,} ({flops_reduction:+.1f}%)")
    print(f"Memory: {fp32_profile.total_memory_access:,} → {quant_profile.total_memory_access:,} ({memory_reduction:+.1f}%)")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Performance Profiler for AI Models",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("model", help="Path to ONNX model file")
    parser.add_argument("-o", "--output", default=None, help="Output HTML file path")
    parser.add_argument("-t", "--title", default="Performance Profile", help="Report title")
    parser.add_argument("--summary", action="store_true", help="Print summary only")
    parser.add_argument("--compare-quantized", metavar="QUANT_MODEL", 
                       help="Compare with quantized model")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found!")
        sys.exit(1)
    
    if args.compare_quantized:
        if not os.path.exists(args.compare_quantized):
            print(f"Error: Quantized model '{args.compare_quantized}' not found!")
            sys.exit(1)
        compare_models(args.model, args.compare_quantized)
    else:
        profile_model(args.model, args.output, args.title, args.summary)


if __name__ == "__main__":
    main()
