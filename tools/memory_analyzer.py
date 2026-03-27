"""
内存占用分析工具 (Memory Analyzer)

功能：
  1. 分析模型权重内存占用
  2. 分析中间激活内存占用
  3. 计算峰值内存需求
  4. 内存优化建议
  5. 支持量化模型内存评估
  6. 生成内存分析报告

用法示例：
  python memory_analyzer.py model.onnx -o memory_report.html
  python memory_analyzer.py model.onnx --compare-quantized quantized_model.onnx
  python memory_analyzer.py model.onnx --target arm_neon
"""
import sys
import os
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from my_ai_compiler.frontend.onnx_parser import parse_onnx
from my_ai_compiler.frontend.graph_ir import Graph, Node, Tensor
from my_ai_compiler.backend.core.shape_infer import ShapeInferer
from my_ai_compiler.backend.core.allocator import BufferAllocator


@dataclass
class TensorMemoryInfo:
    """张量内存信息"""
    name: str
    shape: List[int]
    dtype: str
    size_bytes: int
    is_weight: bool
    is_input: bool
    is_output: bool
    is_activation: bool
    lifetime_start: int = -1
    lifetime_end: int = -1


@dataclass
class MemorySnapshot:
    """内存快照"""
    layer_index: int
    layer_name: str
    allocated: int
    freed: int
    peak: int
    active_tensors: List[str]


@dataclass
class MemoryProfile:
    """内存分析结果"""
    total_weights: int
    total_activations: int
    peak_memory: int
    weight_tensors: List[TensorMemoryInfo]
    activation_tensors: List[TensorMemoryInfo]
    memory_timeline: List[MemorySnapshot]
    optimization_hints: List[str]
    dtype_breakdown: Dict[str, int]
    
    @property
    def total_memory(self) -> int:
        return self.total_weights + self.total_activations


class MemoryAnalyzer:
    """内存分析器"""
    
    DTYPE_SIZE = {
        'float32': 4,
        'float16': 2,
        'bfloat16': 2,
        'int8': 1,
        'int16': 2,
        'int32': 4,
        'int64': 8,
        'uint8': 1,
        'bool': 1,
    }
    
    def __init__(self, graph: Graph):
        self.graph = graph
        self.shape_inferer = ShapeInferer(graph)
        self.allocator = BufferAllocator(graph)
    
    def _prod(self, shape: List[int]) -> int:
        r = 1
        for s in shape:
            r *= max(s, 1)
        return r
    
    def _get_dtype_size(self, dtype: str) -> int:
        """获取数据类型大小"""
        return self.DTYPE_SIZE.get(dtype, 4)
    
    def analyze(self) -> MemoryProfile:
        """执行内存分析"""
        weight_tensors = self._analyze_weights()
        activation_tensors = self._analyze_activations()
        
        total_weights = sum(t.size_bytes for t in weight_tensors)
        total_activations = sum(t.size_bytes for t in activation_tensors)
        
        memory_timeline = self._build_memory_timeline(activation_tensors)
        peak_memory = max((s.peak for s in memory_timeline), default=0) + total_weights
        
        optimization_hints = self._generate_optimization_hints(
            weight_tensors, activation_tensors, peak_memory
        )
        
        dtype_breakdown = self._compute_dtype_breakdown(weight_tensors, activation_tensors)
        
        return MemoryProfile(
            total_weights=total_weights,
            total_activations=total_activations,
            peak_memory=peak_memory,
            weight_tensors=weight_tensors,
            activation_tensors=activation_tensors,
            memory_timeline=memory_timeline,
            optimization_hints=optimization_hints,
            dtype_breakdown=dtype_breakdown
        )
    
    def _analyze_weights(self) -> List[TensorMemoryInfo]:
        """分析权重张量"""
        weights = []
        
        for name, tensor in self.graph.tensors.items():
            if tensor.data is not None:
                shape = tensor.shape if tensor.shape else []
                dtype = tensor.dtype if tensor.dtype else 'float32'
                size_bytes = self._prod(shape) * self._get_dtype_size(dtype)
                
                weights.append(TensorMemoryInfo(
                    name=name,
                    shape=shape,
                    dtype=dtype,
                    size_bytes=size_bytes,
                    is_weight=True,
                    is_input=False,
                    is_output=False,
                    is_activation=False
                ))
        
        return sorted(weights, key=lambda x: -x.size_bytes)
    
    def _analyze_activations(self) -> List[TensorMemoryInfo]:
        """分析激活张量"""
        activations = []
        
        input_set = set(self.graph.inputs)
        output_set = set(self.graph.outputs)
        weight_set = set(name for name, t in self.graph.tensors.items() if t.data is not None)
        
        tensor_lifetimes = self._compute_tensor_lifetimes()
        
        for name, tensor in self.graph.tensors.items():
            if name in weight_set:
                continue
            
            shape = self.shape_inferer.get(name)
            if not shape:
                shape = tensor.shape if tensor.shape else []
            
            dtype = tensor.dtype if tensor.dtype else 'float32'
            size_bytes = self._prod(shape) * self._get_dtype_size(dtype)
            
            is_input = name in input_set
            is_output = name in output_set
            
            lifetime = tensor_lifetimes.get(name, (-1, -1))
            
            activations.append(TensorMemoryInfo(
                name=name,
                shape=shape,
                dtype=dtype,
                size_bytes=size_bytes,
                is_weight=False,
                is_input=is_input,
                is_output=is_output,
                is_activation=not (is_input or is_output),
                lifetime_start=lifetime[0],
                lifetime_end=lifetime[1]
            ))
        
        return sorted(activations, key=lambda x: -x.size_bytes)
    
    def _compute_tensor_lifetimes(self) -> Dict[str, Tuple[int, int]]:
        """计算张量生命周期"""
        lifetimes = {}
        
        for i, node in enumerate(self.graph.nodes):
            for inp in node.inputs:
                if inp not in lifetimes:
                    lifetimes[inp] = [i, i]
                else:
                    lifetimes[inp][1] = max(lifetimes[inp][1], i)
            
            for out in node.outputs:
                if out not in lifetimes:
                    lifetimes[out] = [i, i]
                else:
                    lifetimes[out][1] = max(lifetimes[out][1], i)
        
        return {k: tuple(v) for k, v in lifetimes.items()}
    
    def _build_memory_timeline(self, activations: List[TensorMemoryInfo]) -> List[MemorySnapshot]:
        """构建内存时间线"""
        timeline = []
        
        tensor_map = {t.name: t for t in activations}
        
        active_tensors: Set[str] = set()
        current_memory = 0
        peak_memory = 0
        
        for i, node in enumerate(self.graph.nodes):
            allocated = 0
            freed = 0
            
            for out in node.outputs:
                if out in tensor_map:
                    t = tensor_map[out]
                    if t.lifetime_start == i:
                        active_tensors.add(out)
                        allocated += t.size_bytes
                        current_memory += t.size_bytes
            
            for inp in node.inputs:
                if inp in tensor_map:
                    t = tensor_map[inp]
                    if t.lifetime_end == i and inp in active_tensors:
                        active_tensors.remove(inp)
                        freed += t.size_bytes
                        current_memory -= t.size_bytes
            
            peak_memory = max(peak_memory, current_memory)
            
            timeline.append(MemorySnapshot(
                layer_index=i,
                layer_name=node.name,
                allocated=allocated,
                freed=freed,
                peak=current_memory,
                active_tensors=list(active_tensors)
            ))
        
        return timeline
    
    def _generate_optimization_hints(self, weights: List[TensorMemoryInfo],
                                     activations: List[TensorMemoryInfo],
                                     peak_memory: int) -> List[str]:
        """生成优化建议"""
        hints = []
        
        large_weights = [w for w in weights if w.size_bytes > 10 * 1024 * 1024]
        if large_weights:
            hints.append(f"发现 {len(large_weights)} 个大权重张量(>10MB)，考虑使用INT8量化减少内存占用")
        
        large_activations = [a for a in activations if a.size_bytes > 5 * 1024 * 1024]
        if large_activations:
            hints.append(f"发现 {len(large_activations)} 个大激活张量(>5MB)，考虑使用更小的batch size或特征图压缩")
        
        if peak_memory > 500 * 1024 * 1024:
            hints.append("峰值内存超过500MB，建议考虑梯度检查点或模型并行策略")
        
        int8_weights = [w for w in weights if w.dtype == 'int8']
        if not int8_weights and weights:
            hints.append("当前未使用量化，启用INT8量化可减少约75%的权重内存")
        
        float16_activations = [a for a in activations if a.dtype == 'float16']
        if not float16_activations and activations:
            hints.append("考虑使用FP16混合精度训练/推理，可减少约50%的激活内存")
        
        return hints
    
    def _compute_dtype_breakdown(self, weights: List[TensorMemoryInfo],
                                 activations: List[TensorMemoryInfo]) -> Dict[str, int]:
        """计算各数据类型内存分布"""
        breakdown = defaultdict(int)
        
        for t in weights:
            breakdown[f"weights_{t.dtype}"] += t.size_bytes
        
        for t in activations:
            breakdown[f"activations_{t.dtype}"] += t.size_bytes
        
        return dict(breakdown)
    
    def generate_html_report(self, profile: MemoryProfile, output_path: str,
                            title: str = "Memory Analysis") -> None:
        """生成HTML内存报告"""
        html = self._generate_memory_html(profile, title)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"[MemoryAnalyzer] Memory report saved to: {output_path}")
    
    def _generate_memory_html(self, profile: MemoryProfile, title: str) -> str:
        """生成内存分析HTML"""
        
        weight_rows = []
        for i, t in enumerate(profile.weight_tensors[:50]):
            weight_rows.append(f'''
                <tr>
                    <td>{i}</td>
                    <td style="font-family: monospace; font-size: 11px;">{t.name[:40]}</td>
                    <td>{t.dtype}</td>
                    <td>{str(t.shape)[:30]}</td>
                    <td>{self._format_bytes(t.size_bytes)}</td>
                    <td>{t.size_bytes / profile.total_weights * 100:.1f}%</td>
                </tr>
            ''')
        
        activation_rows = []
        for i, t in enumerate(profile.activation_tensors[:50]):
            activation_rows.append(f'''
                <tr>
                    <td>{i}</td>
                    <td style="font-family: monospace; font-size: 11px;">{t.name[:40]}</td>
                    <td>{t.dtype}</td>
                    <td>{str(t.shape)[:30]}</td>
                    <td>{self._format_bytes(t.size_bytes)}</td>
                    <td>{t.lifetime_start} - {t.lifetime_end}</td>
                </tr>
            ''')
        
        timeline_data = json.dumps([
            {
                'layer': s.layer_name[:20],
                'peak': s.peak,
                'allocated': s.allocated,
                'freed': s.freed
            }
            for s in profile.memory_timeline
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
            background: linear-gradient(135deg, #4776E6 0%, #8E54E9 100%);
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
        .stat-card .value {{ font-size: 24px; font-weight: bold; color: #4776E6; }}
        
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
            border-bottom: 2px solid #4776E6;
        }}
        
        .hint-box {{
            background: #e3f2fd;
            border-left: 4px solid #2196F3;
            padding: 15px;
            margin-bottom: 10px;
            border-radius: 4px;
        }}
        .hint-box.warning {{
            background: #fff3e0;
            border-left-color: #ff9800;
        }}
        
        .memory-bar {{
            display: flex;
            height: 40px;
            border-radius: 8px;
            overflow: hidden;
            margin: 10px 0;
        }}
        .memory-segment {{
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 12px;
            font-weight: bold;
        }}
        .segment-weights {{ background: #4776E6; }}
        .segment-activations {{ background: #8E54E9; }}
        
        .table-container {{
            max-height: 400px;
            overflow-y: auto;
        }}
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}
        .data-table th {{
            background: #f8f9fa;
            padding: 12px;
            text-align: left;
            border-bottom: 2px solid #dee2e6;
            position: sticky;
            top: 0;
        }}
        .data-table td {{
            padding: 10px 12px;
            border-bottom: 1px solid #eee;
        }}
        .data-table tr:hover {{ background: #f8f9fa; }}
        
        .timeline-chart {{
            height: 300px;
            position: relative;
            background: #fafafa;
            border-radius: 8px;
            padding: 20px;
        }}
        .timeline-bar {{
            position: absolute;
            bottom: 40px;
            width: 8px;
            background: linear-gradient(to top, #4776E6, #8E54E9);
            border-radius: 2px;
            transition: all 0.2s;
        }}
        .timeline-bar:hover {{
            background: #ff9800;
            transform: scaleY(1.1);
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>💾 {title}</h1>
            <p>Memory Usage Analysis Report</p>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <h3>Total Memory</h3>
                <div class="value">{self._format_bytes(profile.total_memory)}</div>
            </div>
            <div class="stat-card">
                <h3>Weights</h3>
                <div class="value">{self._format_bytes(profile.total_weights)}</div>
            </div>
            <div class="stat-card">
                <h3>Activations</h3>
                <div class="value">{self._format_bytes(profile.total_activations)}</div>
            </div>
            <div class="stat-card">
                <h3>Peak Memory</h3>
                <div class="value">{self._format_bytes(profile.peak_memory)}</div>
            </div>
        </div>
        
        <div class="section">
            <h2>📊 Memory Distribution</h2>
            <div class="memory-bar">
                <div class="memory-segment segment-weights" 
                     style="width: {profile.total_weights / profile.total_memory * 100}%;">
                    Weights
                </div>
                <div class="memory-segment segment-activations" 
                     style="width: {profile.total_activations / profile.total_memory * 100}%;">
                    Activations
                </div>
            </div>
            <p style="text-align: center; margin-top: 10px; color: #666;">
                Weights: {self._format_bytes(profile.total_weights)} ({profile.total_weights / profile.total_memory * 100:.1f}%) | 
                Activations: {self._format_bytes(profile.total_activations)} ({profile.total_activations / profile.total_memory * 100:.1f}%)
            </p>
        </div>
        
        <div class="section">
            <h2>💡 Optimization Hints</h2>
            {''.join(f'<div class="hint-box">{hint}</div>' for hint in profile.optimization_hints)}
            {'' if profile.optimization_hints else '<div class="hint-box">No optimization hints available.</div>'}
        </div>
        
        <div class="section">
            <h2>📁 Weight Tensors (Top 50)</h2>
            <div class="table-container">
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>Name</th>
                            <th>DType</th>
                            <th>Shape</th>
                            <th>Size</th>
                            <th>% Total</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(weight_rows)}
                    </tbody>
                </table>
            </div>
        </div>
        
        <div class="section">
            <h2>🔄 Activation Tensors (Top 50)</h2>
            <div class="table-container">
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>Name</th>
                            <th>DType</th>
                            <th>Shape</th>
                            <th>Size</th>
                            <th>Lifetime</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(activation_rows)}
                    </tbody>
                </table>
            </div>
        </div>
        
        <div class="section">
            <h2>📈 Memory Timeline</h2>
            <div class="timeline-chart" id="timeline-chart"></div>
        </div>
    </div>
    
    <script>
        const timelineData = {timeline_data};
        const container = document.getElementById('timeline-chart');
        const width = container.offsetWidth - 40;
        const height = container.offsetHeight - 60;
        
        const maxPeak = Math.max(...timelineData.map(d => d.peak)) || 1;
        const barWidth = Math.max(2, width / timelineData.length - 1);
        
        timelineData.forEach((d, i) => {{
            const bar = document.createElement('div');
            bar.className = 'timeline-bar';
            bar.style.left = `${{20 + i * (barWidth + 1)}}px`;
            bar.style.height = `${{(d.peak / maxPeak) * height}}px`;
            bar.title = `${{d.layer}}\\nPeak: ${{(d.peak / 1024 / 1024).toFixed(2)}} MB`;
            container.appendChild(bar);
        }});
    </script>
</body>
</html>'''
    
    def _format_bytes(self, size: int) -> str:
        """格式化字节大小"""
        if size >= 1e9:
            return f"{size/1e9:.2f} GB"
        elif size >= 1e6:
            return f"{size/1e6:.2f} MB"
        elif size >= 1e3:
            return f"{size/1e3:.2f} KB"
        return f"{size} B"
    
    def print_summary(self, profile: MemoryProfile) -> None:
        """打印内存摘要"""
        print("\n" + "="*60)
        print("💾 Memory Analysis Summary")
        print("="*60)
        print(f"Total Memory: {self._format_bytes(profile.total_memory)}")
        print(f"  Weights: {self._format_bytes(profile.total_weights)}")
        print(f"  Activations: {self._format_bytes(profile.total_activations)}")
        print(f"Peak Memory: {self._format_bytes(profile.peak_memory)}")
        
        print("\n📊 Data Type Breakdown:")
        for dtype, size in sorted(profile.dtype_breakdown.items(), key=lambda x: -x[1]):
            print(f"  {dtype:20s}: {self._format_bytes(size)}")
        
        print("\n💡 Optimization Hints:")
        for hint in profile.optimization_hints:
            print(f"  • {hint}")
        
        print("\n" + "="*60)


def analyze_memory(model_path: str, output_path: Optional[str] = None,
                   title: str = "Memory Analysis", summary_only: bool = False) -> MemoryProfile:
    """分析模型内存"""
    print(f"\n[MemoryAnalyzer] Loading model: {model_path}")
    graph = parse_onnx(model_path)
    
    analyzer = MemoryAnalyzer(graph)
    profile = analyzer.analyze()
    
    if summary_only:
        analyzer.print_summary(profile)
    else:
        if output_path is None:
            output_path = model_path.replace('.onnx', '_memory.html')
        analyzer.generate_html_report(profile, output_path, title)
        analyzer.print_summary(profile)
    
    return profile


def compare_memory(fp32_path: str, quant_path: str) -> None:
    """比较量化前后的内存占用"""
    print("\n" + "="*60)
    print("📊 Comparing FP32 vs Quantized Memory")
    print("="*60)
    
    fp32_profile = analyze_memory(fp32_path, summary_only=True)
    quant_profile = analyze_memory(quant_path, summary_only=True)
    
    print("\n📈 Comparison Results:")
    print("-" * 40)
    
    weight_reduction = (1 - quant_profile.total_weights / fp32_profile.total_weights) * 100
    activation_reduction = (1 - quant_profile.total_activations / fp32_profile.total_activations) * 100
    peak_reduction = (1 - quant_profile.peak_memory / fp32_profile.peak_memory) * 100
    
    print(f"Weights: {fp32_profile.total_weights:,} → {quant_profile.total_weights:,} bytes ({weight_reduction:+.1f}%)")
    print(f"Activations: {fp32_profile.total_activations:,} → {quant_profile.total_activations:,} bytes ({activation_reduction:+.1f}%)")
    print(f"Peak Memory: {fp32_profile.peak_memory:,} → {quant_profile.peak_memory:,} bytes ({peak_reduction:+.1f}%)")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Memory Analyzer for AI Models",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("model", help="Path to ONNX model file")
    parser.add_argument("-o", "--output", default=None, help="Output HTML file path")
    parser.add_argument("-t", "--title", default="Memory Analysis", help="Report title")
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
        compare_memory(args.model, args.compare_quantized)
    else:
        analyze_memory(args.model, args.output, args.title, args.summary)


if __name__ == "__main__":
    main()
