"""
统一分析工具 (Unified Analysis Tool)

整合所有分析功能的统一入口：
  1. 模型结构可视化
  2. 性能分析
  3. 内存占用分析
  4. 综合报告生成

用法示例：
  python analyze.py model.onnx --all -o report.html
  python analyze.py model.onnx --structure --performance
  python analyze.py model.onnx --compare quantized_model.onnx
"""
import sys
import os
import argparse
from pathlib import Path
from typing import Optional
import json

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from my_ai_compiler.frontend.onnx_parser import parse_onnx
from my_ai_compiler.frontend.graph_ir import Graph
from my_ai_compiler.tools.model_visualizer import ModelVisualizer
from my_ai_compiler.tools.performance_profiler import PerformanceProfiler, ModelProfile
from my_ai_compiler.tools.memory_analyzer import MemoryAnalyzer, MemoryProfile


class UnifiedAnalyzer:
    """统一分析器"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.graph: Optional[Graph] = None
        self.visualizer: Optional[ModelVisualizer] = None
        self.profiler: Optional[PerformanceProfiler] = None
        self.memory_analyzer: Optional[MemoryAnalyzer] = None
        self.profile: Optional[ModelProfile] = None
        self.memory_profile: Optional[MemoryProfile] = None
    
    def load_model(self) -> None:
        """加载模型"""
        print(f"\n[Analyzer] Loading model: {self.model_path}")
        self.graph = parse_onnx(self.model_path)
        print(f"[Analyzer] Model loaded: {len(self.graph.nodes)} nodes, {len(self.graph.tensors)} tensors")
    
    def analyze_structure(self, output_path: Optional[str] = None) -> None:
        """分析模型结构"""
        if self.graph is None:
            self.load_model()
        
        print("\n[Analyzer] Analyzing model structure...")
        self.visualizer = ModelVisualizer(self.graph)
        
        if output_path:
            self.visualizer.generate_html_report(output_path, "Model Structure Analysis")
        else:
            self.visualizer.print_summary()
    
    def analyze_performance(self, output_path: Optional[str] = None) -> ModelProfile:
        """分析性能"""
        if self.graph is None:
            self.load_model()
        
        print("\n[Analyzer] Analyzing performance...")
        self.profiler = PerformanceProfiler(self.graph)
        self.profile = self.profiler.profile()
        
        if output_path:
            self.profiler.generate_html_report(self.profile, output_path, "Performance Analysis")
        else:
            self.profiler.print_summary(self.profile)
        
        return self.profile
    
    def analyze_memory(self, output_path: Optional[str] = None) -> MemoryProfile:
        """分析内存"""
        if self.graph is None:
            self.load_model()
        
        print("\n[Analyzer] Analyzing memory usage...")
        self.memory_analyzer = MemoryAnalyzer(self.graph)
        self.memory_profile = self.memory_analyzer.analyze()
        
        if output_path:
            self.memory_analyzer.generate_html_report(self.memory_profile, output_path, "Memory Analysis")
        else:
            self.memory_analyzer.print_summary(self.memory_profile)
        
        return self.memory_profile
    
    def generate_comprehensive_report(self, output_path: str) -> None:
        """生成综合分析报告"""
        if self.graph is None:
            self.load_model()
        
        if self.profile is None:
            self.analyze_performance()
        
        if self.memory_profile is None:
            self.analyze_memory()
        
        print(f"\n[Analyzer] Generating comprehensive report: {output_path}")
        
        html = self._generate_comprehensive_html()
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"[Analyzer] Comprehensive report saved to: {output_path}")
    
    def _generate_comprehensive_html(self) -> str:
        """生成综合报告HTML"""
        
        operator_stats = self.visualizer.get_operator_stats() if self.visualizer else {}
        
        return f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Comprehensive Model Analysis</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f7fa;
            color: #333;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
        .header {{ 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .header h1 {{ font-size: 32px; margin-bottom: 10px; }}
        .header p {{ opacity: 0.9; font-size: 16px; }}
        
        .nav {{
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }}
        .nav-btn {{
            padding: 10px 20px;
            background: white;
            border: 2px solid #667eea;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.3s;
        }}
        .nav-btn:hover {{
            background: #667eea;
            color: white;
        }}
        .nav-btn.active {{
            background: #667eea;
            color: white;
        }}
        
        .tab-content {{ display: none; }}
        .tab-content.active {{ display: block; }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .stat-card h3 {{ color: #666; font-size: 13px; margin-bottom: 8px; text-transform: uppercase; }}
        .stat-card .value {{ font-size: 28px; font-weight: bold; }}
        .stat-card .unit {{ font-size: 12px; color: #999; }}
        
        .section {{
            background: white;
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .section h2 {{ 
            font-size: 20px; 
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .two-col {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        @media (max-width: 900px) {{
            .two-col {{ grid-template-columns: 1fr; }}
        }}
        
        .progress-bar {{
            height: 24px;
            background: #e0e0e0;
            border-radius: 12px;
            overflow: hidden;
            margin: 10px 0;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            display: flex;
            align-items: center;
            padding-left: 10px;
            color: white;
            font-size: 12px;
            font-weight: bold;
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
        }}
        .data-table td {{
            padding: 10px 12px;
            border-bottom: 1px solid #eee;
        }}
        .data-table tr:hover {{ background: #f8f9fa; }}
        
        .table-container {{
            max-height: 400px;
            overflow-y: auto;
        }}
        
        .badge {{
            display: inline-block;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
            color: white;
        }}
        .badge-conv {{ background: #4CAF50; }}
        .badge-pool {{ background: #2196F3; }}
        .badge-activation {{ background: #FF9800; }}
        .badge-norm {{ background: #9C27B0; }}
        .badge-linear {{ background: #F44336; }}
        .badge-other {{ background: #607D8B; }}
        
        .metric-row {{
            display: flex;
            justify-content: space-between;
            padding: 12px 0;
            border-bottom: 1px solid #eee;
        }}
        .metric-label {{ color: #666; }}
        .metric-value {{ font-weight: 600; }}
        
        .chart-placeholder {{
            height: 200px;
            background: #f8f9fa;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #999;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔬 Comprehensive Model Analysis</h1>
            <p>{self.model_path}</p>
        </div>
        
        <div class="nav">
            <button class="nav-btn active" onclick="showTab('overview')">📊 Overview</button>
            <button class="nav-btn" onclick="showTab('structure')">🔗 Structure</button>
            <button class="nav-btn" onclick="showTab('performance')">⚡ Performance</button>
            <button class="nav-btn" onclick="showTab('memory')">💾 Memory</button>
        </div>
        
        <!-- Overview Tab -->
        <div id="overview" class="tab-content active">
            <div class="stats-grid">
                <div class="stat-card">
                    <h3>Total Layers</h3>
                    <div class="value">{len(self.graph.nodes)}</div>
                </div>
                <div class="stat-card">
                    <h3>Operator Types</h3>
                    <div class="value">{len(operator_stats)}</div>
                </div>
                <div class="stat-card">
                    <h3>Total FLOPs</h3>
                    <div class="value">{self._format_flops(self.profile.total_flops)}</div>
                </div>
                <div class="stat-card">
                    <h3>Total Parameters</h3>
                    <div class="value">{self._format_memory(self.profile.total_params)}</div>
                </div>
                <div class="stat-card">
                    <h3>Peak Memory</h3>
                    <div class="value">{self._format_memory(self.memory_profile.peak_memory)}</div>
                </div>
                <div class="stat-card">
                    <h3>Memory Bound</h3>
                    <div class="value">{self.profile.memory_bound_ratio*100:.0f}%</div>
                </div>
            </div>
            
            <div class="two-col">
                <div class="section">
                    <h2>📊 Operator Distribution</h2>
                    {self._generate_operator_distribution_html(operator_stats)}
                </div>
                
                <div class="section">
                    <h2>💡 Key Metrics</h2>
                    <div class="metric-row">
                        <span class="metric-label">Model Name</span>
                        <span class="metric-value">{self.graph.name}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Total Tensors</span>
                        <span class="metric-value">{len(self.graph.tensors)}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Input Tensors</span>
                        <span class="metric-value">{len(self.graph.inputs)}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Output Tensors</span>
                        <span class="metric-value">{len(self.graph.outputs)}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Weights Memory</span>
                        <span class="metric-value">{self._format_memory(self.memory_profile.total_weights)}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Activations Memory</span>
                        <span class="metric-value">{self._format_memory(self.memory_profile.total_activations)}</span>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Structure Tab -->
        <div id="structure" class="tab-content">
            <div class="section">
                <h2>🔗 Layer Structure</h2>
                <div class="table-container">
                    <table class="data-table">
                        <thead>
                            <tr>
                                <th>#</th>
                                <th>Name</th>
                                <th>Op Type</th>
                                <th>Inputs</th>
                                <th>Outputs</th>
                            </tr>
                        </thead>
                        <tbody>
                            {self._generate_layer_table_html()}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        
        <!-- Performance Tab -->
        <div id="performance" class="tab-content">
            <div class="stats-grid">
                <div class="stat-card">
                    <h3>Total FLOPs</h3>
                    <div class="value">{self._format_flops(self.profile.total_flops)}</div>
                    <div class="unit">{self.profile.total_flops:,} ops</div>
                </div>
                <div class="stat-card">
                    <h3>Memory Access</h3>
                    <div class="value">{self._format_memory(self.profile.total_memory_access)}</div>
                </div>
                <div class="stat-card">
                    <h3>Parameters</h3>
                    <div class="value">{self._format_memory(self.profile.total_params)}</div>
                </div>
                <div class="stat-card">
                    <h3>Memory Bound Ratio</h3>
                    <div class="value">{self.profile.memory_bound_ratio*100:.1f}%</div>
                </div>
            </div>
            
            <div class="section">
                <h2>🔥 Performance Bottlenecks</h2>
                <div style="display: flex; flex-wrap: wrap; gap: 10px;">
                    {''.join(f'<span class="badge badge-other">{name}</span>' for name in self.profile.bottleneck_layers)}
                </div>
            </div>
            
            <div class="section">
                <h2>📊 FLOPs Distribution</h2>
                {self._generate_flops_distribution_html()}
            </div>
        </div>
        
        <!-- Memory Tab -->
        <div id="memory" class="tab-content">
            <div class="stats-grid">
                <div class="stat-card">
                    <h3>Total Memory</h3>
                    <div class="value">{self._format_memory(self.memory_profile.total_memory)}</div>
                </div>
                <div class="stat-card">
                    <h3>Weights</h3>
                    <div class="value">{self._format_memory(self.memory_profile.total_weights)}</div>
                </div>
                <div class="stat-card">
                    <h3>Activations</h3>
                    <div class="value">{self._format_memory(self.memory_profile.total_activations)}</div>
                </div>
                <div class="stat-card">
                    <h3>Peak Memory</h3>
                    <div class="value">{self._format_memory(self.memory_profile.peak_memory)}</div>
                </div>
            </div>
            
            <div class="section">
                <h2>💡 Optimization Hints</h2>
                {''.join(f'<div class="metric-row"><span>{hint}</span></div>' for hint in self.memory_profile.optimization_hints)}
            </div>
            
            <div class="section">
                <h2>📁 Top Weight Tensors</h2>
                <div class="table-container">
                    <table class="data-table">
                        <thead>
                            <tr>
                                <th>Name</th>
                                <th>Shape</th>
                                <th>Size</th>
                            </tr>
                        </thead>
                        <tbody>
                            {self._generate_weights_table_html()}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        function showTab(tabId) {{
            document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.nav-btn').forEach(el => el.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');
            event.target.classList.add('active');
        }}
    </script>
</body>
</html>'''
    
    def _generate_operator_distribution_html(self, stats: dict) -> str:
        """生成算子分布HTML"""
        if not stats:
            return '<div class="chart-placeholder">No data</div>'
        
        max_count = max(stats.values())
        rows = []
        for op, count in list(stats.items())[:8]:
            pct = count / max_count * 100
            rows.append(f'''
                <div style="margin-bottom: 8px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                        <span>{op}</span>
                        <span>{count}</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {pct}%;"></div>
                    </div>
                </div>
            ''')
        return '\n'.join(rows)
    
    def _generate_layer_table_html(self) -> str:
        """生成层表格HTML"""
        rows = []
        for i, node in enumerate(self.graph.nodes):
            badge_class = self._get_badge_class(node.op_type)
            rows.append(f'''
                <tr>
                    <td>{i}</td>
                    <td style="font-family: monospace; font-size: 11px;">{node.name[:30]}</td>
                    <td><span class="badge {badge_class}">{node.op_type}</span></td>
                    <td>{len(node.inputs)}</td>
                    <td>{len(node.outputs)}</td>
                </tr>
            ''')
        return '\n'.join(rows)
    
    def _generate_flops_distribution_html(self) -> str:
        """生成FLOPs分布HTML"""
        if not self.profile.operator_breakdown:
            return '<div class="chart-placeholder">No data</div>'
        
        total = self.profile.total_flops
        rows = []
        for op, flops in list(self.profile.operator_breakdown.items())[:8]:
            pct = flops / total * 100 if total > 0 else 0
            rows.append(f'''
                <div style="margin-bottom: 8px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                        <span>{op}</span>
                        <span>{self._format_flops(flops)} ({pct:.1f}%)</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {pct}%;"></div>
                    </div>
                </div>
            ''')
        return '\n'.join(rows)
    
    def _generate_weights_table_html(self) -> str:
        """生成权重表格HTML"""
        rows = []
        for t in self.memory_profile.weight_tensors[:20]:
            rows.append(f'''
                <tr>
                    <td style="font-family: monospace; font-size: 11px;">{t.name[:40]}</td>
                    <td>{str(t.shape)[:30]}</td>
                    <td>{self._format_memory(t.size_bytes)}</td>
                </tr>
            ''')
        return '\n'.join(rows)
    
    def _get_badge_class(self, op_type: str) -> str:
        """获取徽章样式类"""
        if 'Conv' in op_type:
            return 'badge-conv'
        elif 'Pool' in op_type:
            return 'badge-pool'
        elif op_type in ['Relu', 'Clip', 'Sigmoid', 'Tanh']:
            return 'badge-activation'
        elif 'Norm' in op_type:
            return 'badge-norm'
        elif op_type in ['Gemm', 'MatMul', 'Linear']:
            return 'badge-linear'
        return 'badge-other'
    
    def _format_flops(self, flops: int) -> str:
        if flops >= 1e9:
            return f"{flops/1e9:.2f}G"
        elif flops >= 1e6:
            return f"{flops/1e6:.2f}M"
        elif flops >= 1e3:
            return f"{flops/1e3:.2f}K"
        return str(flops)
    
    def _format_memory(self, size: int) -> str:
        size *= 4
        if size >= 1e9:
            return f"{size/1e9:.2f} GB"
        elif size >= 1e6:
            return f"{size/1e6:.2f} MB"
        elif size >= 1e3:
            return f"{size/1e3:.2f} KB"
        return f"{size} B"


def analyze_model(model_path: str, output_path: Optional[str] = None,
                  structure: bool = False, performance: bool = False,
                  memory: bool = False, all_analysis: bool = False,
                  summary_only: bool = False) -> None:
    """分析模型"""
    analyzer = UnifiedAnalyzer(model_path)
    
    if all_analysis or (not structure and not performance and not memory):
        if output_path and not summary_only:
            analyzer.generate_comprehensive_report(output_path)
        else:
            analyzer.load_model()
            if structure or all_analysis:
                analyzer.analyze_structure()
            if performance or all_analysis:
                analyzer.analyze_performance()
            if memory or all_analysis:
                analyzer.analyze_memory()
    else:
        if structure:
            struct_output = output_path.replace('.html', '_structure.html') if output_path else None
            analyzer.analyze_structure(struct_output)
        
        if performance:
            perf_output = output_path.replace('.html', '_performance.html') if output_path else None
            analyzer.analyze_performance(perf_output)
        
        if memory:
            mem_output = output_path.replace('.html', '_memory.html') if output_path else None
            analyzer.analyze_memory(mem_output)


def compare_models(fp32_path: str, quant_path: str, output_path: Optional[str] = None) -> None:
    """比较两个模型"""
    print("\n" + "="*60)
    print("📊 Comparing Models")
    print("="*60)
    
    print("\n[1/2] Analyzing FP32 model:")
    fp32_analyzer = UnifiedAnalyzer(fp32_path)
    fp32_profile = fp32_analyzer.analyze_performance()
    fp32_memory = fp32_analyzer.analyze_memory()
    
    print("\n[2/2] Analyzing Quantized model:")
    quant_analyzer = UnifiedAnalyzer(quant_path)
    quant_profile = quant_analyzer.analyze_performance()
    quant_memory = quant_analyzer.analyze_memory()
    
    print("\n" + "="*60)
    print("📈 Comparison Summary")
    print("="*60)
    
    flops_change = (quant_profile.total_flops - fp32_profile.total_flops) / fp32_profile.total_flops * 100
    memory_change = (quant_memory.total_memory - fp32_memory.total_memory) / fp32_memory.total_memory * 100
    peak_change = (quant_memory.peak_memory - fp32_memory.peak_memory) / fp32_memory.peak_memory * 100
    
    print(f"\nFLOPs: {fp32_profile.total_flops:,} → {quant_profile.total_flops:,} ({flops_change:+.1f}%)")
    print(f"Total Memory: {fp32_memory.total_memory:,} → {quant_memory.total_memory:,} bytes ({memory_change:+.1f}%)")
    print(f"Peak Memory: {fp32_memory.peak_memory:,} → {quant_memory.peak_memory:,} bytes ({peak_change:+.1f}%)")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Unified Model Analysis Tool",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("model", help="Path to ONNX model file")
    parser.add_argument("-o", "--output", default=None, help="Output HTML report path")
    parser.add_argument("--structure", action="store_true", help="Analyze model structure")
    parser.add_argument("--performance", action="store_true", help="Analyze performance")
    parser.add_argument("--memory", action="store_true", help="Analyze memory usage")
    parser.add_argument("--all", dest="all_analysis", action="store_true", 
                       help="Run all analyses (default)")
    parser.add_argument("--summary", action="store_true", help="Print summary only (no HTML)")
    parser.add_argument("--compare", metavar="QUANT_MODEL", 
                       help="Compare with another model (e.g., quantized)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found!")
        sys.exit(1)
    
    if args.compare:
        if not os.path.exists(args.compare):
            print(f"Error: Comparison model '{args.compare}' not found!")
            sys.exit(1)
        compare_models(args.model, args.compare, args.output)
    else:
        analyze_model(
            args.model, 
            args.output,
            args.structure,
            args.performance,
            args.memory,
            args.all_analysis,
            args.summary
        )


if __name__ == "__main__":
    main()
