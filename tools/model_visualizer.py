"""
模型结构可视化工具

功能：
  1. 生成计算图可视化（支持多种格式：HTML、SVG、PNG）
  2. 统计算子分布
  3. 分析网络拓扑结构
  4. 导出模型摘要报告

用法示例：
  python model_visualizer.py model.onnx -o report.html
  python model_visualizer.py model.onnx --format svg --output graph.svg
  python model_visualizer.py model.onnx --summary
"""
import sys
import os
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import json

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from my_ai_compiler.frontend.onnx_parser import parse_onnx
from my_ai_compiler.frontend.graph_ir import Graph, Node, Tensor
from my_ai_compiler.backend.core.shape_infer import ShapeInferer
from my_ai_compiler.backend.core.allocator import BufferAllocator


class ModelVisualizer:
    """模型结构可视化器"""
    
    def __init__(self, graph: Graph):
        self.graph = graph
        self.shape_inferer = ShapeInferer(graph)
        self.allocator = BufferAllocator(graph)
        
    def get_operator_stats(self) -> Dict[str, int]:
        """统计各算子出现次数"""
        stats = defaultdict(int)
        for node in self.graph.nodes:
            stats[node.op_type] += 1
        return dict(sorted(stats.items(), key=lambda x: -x[1]))
    
    def get_layer_details(self) -> List[Dict]:
        """获取每层详细信息"""
        details = []
        for i, node in enumerate(self.graph.nodes):
            input_shapes = []
            for inp in node.inputs:
                shape = self.shape_inferer.get(inp)
                input_shapes.append(shape if shape else [])
            
            output_shapes = []
            for out in node.outputs:
                shape = self.shape_inferer.get(out)
                output_shapes.append(shape if shape else [])
            
            attrs_str = ""
            if node.attrs:
                attrs_str = ", ".join(f"{k}={v}" for k, v in list(node.attrs.items())[:5])
                if len(node.attrs) > 5:
                    attrs_str += "..."
            
            details.append({
                'index': i,
                'name': node.name,
                'op_type': node.op_type,
                'inputs': node.inputs,
                'outputs': node.outputs,
                'input_shapes': input_shapes,
                'output_shapes': output_shapes,
                'attrs': node.attrs,
                'attrs_str': attrs_str
            })
        return details
    
    def get_tensor_info(self) -> Dict[str, Dict]:
        """获取所有张量信息"""
        info = {}
        for name, tensor in self.graph.tensors.items():
            info[name] = {
                'shape': tensor.shape,
                'dtype': tensor.dtype,
                'is_weight': tensor.data is not None,
                'size': self._prod(tensor.shape) if tensor.shape else 0
            }
        return info
    
    def _prod(self, shape: List[int]) -> int:
        r = 1
        for s in shape:
            r *= max(s, 1)
        return r
    
    def generate_html_report(self, output_path: str, title: str = "Model Visualization") -> None:
        """生成交互式HTML报告"""
        operator_stats = self.get_operator_stats()
        layer_details = self.get_layer_details()
        tensor_info = self.get_tensor_info()
        
        total_params = sum(t['size'] for t in tensor_info.values() if t['is_weight'])
        total_activations = sum(
            self.allocator.size(name) 
            for name in self.graph.tensors.keys() 
            if not tensor_info.get(name, {}).get('is_weight', False)
        )
        
        html_content = self._generate_html_template(
            title=title,
            graph=self.graph,
            operator_stats=operator_stats,
            layer_details=layer_details,
            tensor_info=tensor_info,
            total_params=total_params,
            total_activations=total_activations
        )
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"[Visualizer] HTML report saved to: {output_path}")
    
    def _generate_html_template(self, title: str, graph: Graph, 
                                 operator_stats: Dict, layer_details: List,
                                 tensor_info: Dict, total_params: int,
                                 total_activations: int) -> str:
        """生成HTML模板"""
        
        nodes_json = json.dumps([
            {
                'id': i,
                'name': d['name'],
                'op_type': d['op_type'],
                'inputs': d['inputs'],
                'outputs': d['outputs'],
                'attrs_str': d['attrs_str']
            }
            for i, d in enumerate(layer_details)
        ])
        
        edges_json = json.dumps(self._build_edges())
        
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
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .header h1 {{ font-size: 28px; margin-bottom: 10px; }}
        .header p {{ opacity: 0.9; }}
        
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
        .stat-card .value {{ font-size: 28px; font-weight: bold; color: #333; }}
        
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
            border-bottom: 2px solid #667eea;
        }}
        
        .chart-container {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }}
        .bar-item {{
            display: flex;
            align-items: center;
            width: 100%;
            margin-bottom: 8px;
        }}
        .bar-label {{ width: 120px; font-size: 13px; }}
        .bar-container {{ 
            flex: 1; 
            background: #e0e0e0; 
            border-radius: 4px; 
            height: 24px;
            overflow: hidden;
        }}
        .bar-fill {{
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            display: flex;
            align-items: center;
            padding-left: 8px;
            color: white;
            font-size: 12px;
            font-weight: bold;
        }}
        
        .graph-container {{
            width: 100%;
            height: 600px;
            border: 1px solid #ddd;
            border-radius: 8px;
            overflow: hidden;
            position: relative;
        }}
        #graph-canvas {{
            width: 100%;
            height: 100%;
            cursor: grab;
        }}
        #graph-canvas:active {{ cursor: grabbing; }}
        
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
            color: white;
        }}
        .op-conv {{ background: #4CAF50; }}
        .op-pool {{ background: #2196F3; }}
        .op-activation {{ background: #FF9800; }}
        .op-norm {{ background: #9C27B0; }}
        .op-linear {{ background: #F44336; }}
        .op-other {{ background: #607D8B; }}
        
        .table-container {{
            max-height: 400px;
            overflow-y: auto;
        }}
        
        .tooltip {{
            position: absolute;
            background: rgba(0,0,0,0.8);
            color: white;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 12px;
            pointer-events: none;
            z-index: 1000;
            display: none;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔬 {title}</h1>
            <p>Model: {graph.name} | Nodes: {len(graph.nodes)} | Tensors: {len(graph.tensors)}</p>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <h3>Total Layers</h3>
                <div class="value">{len(graph.nodes)}</div>
            </div>
            <div class="stat-card">
                <h3>Operator Types</h3>
                <div class="value">{len(operator_stats)}</div>
            </div>
            <div class="stat-card">
                <h3>Total Parameters</h3>
                <div class="value">{self._format_number(total_params)}</div>
            </div>
            <div class="stat-card">
                <h3>Activation Size</h3>
                <div class="value">{self._format_number(total_activations)}</div>
            </div>
        </div>
        
        <div class="section">
            <h2>📊 Operator Distribution</h2>
            <div class="chart-container">
                {self._generate_operator_bars(operator_stats)}
            </div>
        </div>
        
        <div class="section">
            <h2>🔗 Computation Graph</h2>
            <div class="graph-container">
                <canvas id="graph-canvas"></canvas>
            </div>
            <div class="tooltip" id="tooltip"></div>
        </div>
        
        <div class="section">
            <h2>📋 Layer Details</h2>
            <div class="table-container">
                <table class="layer-table">
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>Name</th>
                            <th>Op Type</th>
                            <th>Input Shape</th>
                            <th>Output Shape</th>
                            <th>Attributes</th>
                        </tr>
                    </thead>
                    <tbody>
                        {self._generate_layer_rows(layer_details)}
                    </tbody>
                </table>
            </div>
        </div>
    </div>
    
    <script>
        const nodes = {nodes_json};
        const edges = {edges_json};
        
        const canvas = document.getElementById('graph-canvas');
        const ctx = canvas.getContext('2d');
        const tooltip = document.getElementById('tooltip');
        
        let scale = 1;
        let offsetX = 0;
        let offsetY = 0;
        let isDragging = false;
        let lastX, lastY;
        let nodePositions = [];
        
        function resizeCanvas() {{
            canvas.width = canvas.offsetWidth * 2;
            canvas.height = canvas.offsetHeight * 2;
            ctx.scale(2, 2);
            calculateLayout();
            draw();
        }}
        
        function calculateLayout() {{
            const width = canvas.offsetWidth;
            const height = canvas.offsetHeight;
            const cols = Math.ceil(Math.sqrt(nodes.length));
            const nodeWidth = 120;
            const nodeHeight = 50;
            const hGap = 80;
            const vGap = 60;
            
            nodePositions = nodes.map((node, i) => ({{
                x: 100 + (i % cols) * (nodeWidth + hGap),
                y: 80 + Math.floor(i / cols) * (nodeHeight + vGap),
                width: nodeWidth,
                height: nodeHeight,
                ...node
            }}));
        }}
        
        function getOpColor(opType) {{
            const colors = {{
                'Conv': '#4CAF50', 'ConvBnRelu': '#4CAF50', 'ConvBn': '#4CAF50',
                'MaxPool': '#2196F3', 'AveragePool': '#2196F3', 'GlobalAveragePool': '#2196F3',
                'Relu': '#FF9800', 'Clip': '#FF9800', 'Sigmoid': '#FF9800',
                'BatchNormalization': '#9C27B0', 'InstanceNormalization': '#9C27B0',
                'Gemm': '#F44336', 'MatMul': '#F44336',
            }};
            return colors[opType] || '#607D8B';
        }}
        
        function draw() {{
            const width = canvas.offsetWidth;
            const height = canvas.offsetHeight;
            
            ctx.clearRect(0, 0, width, height);
            ctx.save();
            ctx.translate(offsetX, offsetY);
            ctx.scale(scale, scale);
            
            // Draw edges
            ctx.strokeStyle = '#ccc';
            ctx.lineWidth = 1.5;
            edges.forEach(edge => {{
                const from = nodePositions.find(n => n.id === edge.from);
                const to = nodePositions.find(n => n.id === edge.to);
                if (from && to) {{
                    ctx.beginPath();
                    ctx.moveTo(from.x + from.width/2, from.y + from.height);
                    ctx.bezierCurveTo(
                        from.x + from.width/2, from.y + from.height + 30,
                        to.x + to.width/2, to.y - 30,
                        to.x + to.width/2, to.y
                    );
                    ctx.stroke();
                }}
            }});
            
            // Draw nodes
            nodePositions.forEach(node => {{
                const color = getOpColor(node.op_type);
                
                // Shadow
                ctx.shadowColor = 'rgba(0,0,0,0.2)';
                ctx.shadowBlur = 4;
                ctx.shadowOffsetX = 2;
                ctx.shadowOffsetY = 2;
                
                // Node body
                ctx.fillStyle = color;
                ctx.beginPath();
                ctx.roundRect(node.x, node.y, node.width, node.height, 6);
                ctx.fill();
                
                ctx.shadowColor = 'transparent';
                
                // Node text
                ctx.fillStyle = 'white';
                ctx.font = 'bold 11px sans-serif';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                
                const displayName = node.op_type.length > 12 
                    ? node.op_type.substring(0, 10) + '..' 
                    : node.op_type;
                ctx.fillText(displayName, node.x + node.width/2, node.y + node.height/2 - 6);
                
                ctx.font = '9px sans-serif';
                ctx.fillStyle = 'rgba(255,255,255,0.8)';
                const nameDisplay = node.name.length > 14 
                    ? '..' + node.name.slice(-12) 
                    : node.name;
                ctx.fillText(nameDisplay, node.x + node.width/2, node.y + node.height/2 + 8);
            }});
            
            ctx.restore();
        }}
        
        // Mouse interactions
        canvas.addEventListener('wheel', (e) => {{
            e.preventDefault();
            const delta = e.deltaY > 0 ? 0.9 : 1.1;
            scale = Math.max(0.3, Math.min(3, scale * delta));
            draw();
        }});
        
        canvas.addEventListener('mousedown', (e) => {{
            isDragging = true;
            lastX = e.clientX;
            lastY = e.clientY;
        }});
        
        canvas.addEventListener('mousemove', (e) => {{
            if (isDragging) {{
                offsetX += e.clientX - lastX;
                offsetY += e.clientY - lastY;
                lastX = e.clientX;
                lastY = e.clientY;
                draw();
            }}
            
            // Tooltip
            const rect = canvas.getBoundingClientRect();
            const mx = (e.clientX - rect.left - offsetX) / scale;
            const my = (e.clientY - rect.top - offsetY) / scale;
            
            const hoveredNode = nodePositions.find(n => 
                mx >= n.x && mx <= n.x + n.width &&
                my >= n.y && my <= n.y + n.height
            );
            
            if (hoveredNode) {{
                tooltip.style.display = 'block';
                tooltip.style.left = (e.clientX + 10) + 'px';
                tooltip.style.top = (e.clientY + 10) + 'px';
                tooltip.innerHTML = `<b>${{hoveredNode.op_type}}</b><br>` +
                    `Name: ${{hoveredNode.name}}<br>` +
                    `Inputs: ${{hoveredNode.inputs.join(', ')}}<br>` +
                    `Outputs: ${{hoveredNode.outputs.join(', ')}}`;
            }} else {{
                tooltip.style.display = 'none';
            }}
        }});
        
        canvas.addEventListener('mouseup', () => isDragging = false);
        canvas.addEventListener('mouseleave', () => {{
            isDragging = false;
            tooltip.style.display = 'none';
        }});
        
        window.addEventListener('resize', resizeCanvas);
        resizeCanvas();
    </script>
</body>
</html>'''
    
    def _build_edges(self) -> List[Dict]:
        """构建边信息"""
        edges = []
        node_output_map = {}
        
        for i, node in enumerate(self.graph.nodes):
            for out in node.outputs:
                node_output_map[out] = i
        
        for i, node in enumerate(self.graph.nodes):
            for inp in node.inputs:
                if inp in node_output_map:
                    edges.append({
                        'from': node_output_map[inp],
                        'to': i
                    })
        return edges
    
    def _format_number(self, num: int) -> str:
        """格式化数字"""
        if num >= 1e9:
            return f"{num/1e9:.2f}B"
        elif num >= 1e6:
            return f"{num/1e6:.2f}M"
        elif num >= 1e3:
            return f"{num/1e3:.2f}K"
        return str(num)
    
    def _generate_operator_bars(self, stats: Dict) -> str:
        """生成算子统计条形图"""
        max_count = max(stats.values()) if stats else 1
        bars = []
        for op, count in stats.items():
            width = int(count / max_count * 100)
            bars.append(f'''
                <div class="bar-item">
                    <span class="bar-label">{op}</span>
                    <div class="bar-container">
                        <div class="bar-fill" style="width: {max(width, 5)}%;">{count}</div>
                    </div>
                </div>
            ''')
        return '\n'.join(bars)
    
    def _generate_layer_rows(self, details: List[Dict]) -> str:
        """生成层详情表格行"""
        rows = []
        for d in details:
            op_class = self._get_op_class(d['op_type'])
            input_shape_str = str(d['input_shapes'][0]) if d['input_shapes'] else '-'
            output_shape_str = str(d['output_shapes'][0]) if d['output_shapes'] else '-'
            
            rows.append(f'''
                <tr>
                    <td>{d['index']}</td>
                    <td style="font-family: monospace; font-size: 11px;">{d['name'][:30]}</td>
                    <td><span class="op-badge {op_class}">{d['op_type']}</span></td>
                    <td style="font-family: monospace; font-size: 11px;">{input_shape_str}</td>
                    <td style="font-family: monospace; font-size: 11px;">{output_shape_str}</td>
                    <td style="font-size: 11px; color: #666;">{d['attrs_str']}</td>
                </tr>
            ''')
        return '\n'.join(rows)
    
    def _get_op_class(self, op_type: str) -> str:
        """获取算子样式类"""
        if 'Conv' in op_type:
            return 'op-conv'
        elif 'Pool' in op_type:
            return 'op-pool'
        elif op_type in ['Relu', 'Clip', 'Sigmoid', 'Tanh', 'LeakyRelu']:
            return 'op-activation'
        elif 'Norm' in op_type:
            return 'op-norm'
        elif op_type in ['Gemm', 'MatMul', 'Linear']:
            return 'op-linear'
        return 'op-other'
    
    def print_summary(self) -> None:
        """打印模型摘要到控制台"""
        print("\n" + "="*60)
        print("📊 Model Structure Summary")
        print("="*60)
        print(f"Model Name: {self.graph.name}")
        print(f"Total Nodes: {len(self.graph.nodes)}")
        print(f"Total Tensors: {len(self.graph.tensors)}")
        print(f"Input Tensors: {len(self.graph.inputs)}")
        print(f"Output Tensors: {len(self.graph.outputs)}")
        
        print("\n📈 Operator Distribution:")
        stats = self.get_operator_stats()
        for op, count in list(stats.items())[:10]:
            print(f"  {op:20s}: {count:4d}")
        
        tensor_info = self.get_tensor_info()
        total_params = sum(t['size'] for t in tensor_info.values() if t['is_weight'])
        print(f"\n💾 Total Parameters: {self._format_number(total_params)} ({total_params:,})")
        
        print("\n" + "="*60)


def visualize_model(model_path: str, output_path: Optional[str] = None, 
                    title: str = "Model Visualization", summary_only: bool = False) -> None:
    """可视化模型"""
    print(f"\n[Visualizer] Loading model: {model_path}")
    graph = parse_onnx(model_path)
    
    visualizer = ModelVisualizer(graph)
    
    if summary_only:
        visualizer.print_summary()
    else:
        if output_path is None:
            output_path = model_path.replace('.onnx', '_visualization.html')
        visualizer.generate_html_report(output_path, title)
        visualizer.print_summary()


def main():
    parser = argparse.ArgumentParser(
        description="Model Structure Visualizer",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("model", help="Path to ONNX model file")
    parser.add_argument("-o", "--output", default=None, help="Output HTML file path")
    parser.add_argument("-t", "--title", default="Model Visualization", help="Report title")
    parser.add_argument("--summary", action="store_true", help="Print summary only")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found!")
        sys.exit(1)
    
    visualize_model(args.model, args.output, args.title, args.summary)


if __name__ == "__main__":
    main()
