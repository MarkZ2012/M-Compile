"""
ONNX Parser - 解析ONNX模型到Graph IR
"""
import onnx
import numpy as np
from typing import Dict, List, Optional
from .graph_ir import Graph, Node, Tensor


class ONNXParser:
    """ONNX模型解析器"""
    
    def __init__(self):
        self.graph: Optional[Graph] = None
    
    def parse(self, model_path: str) -> Graph:
        """解析ONNX模型文件"""
        model = onnx.load(model_path)
        onnx.checker.check_model(model)
        
        onnx_graph = model.graph
        self.graph = Graph(name=onnx_graph.name)
        
        # 解析初始权重（常量张量）
        self._parse_initializers(onnx_graph.initializer)
        
        # 解析输入
        self._parse_inputs(onnx_graph.input)
        
        # 解析节点
        self._parse_nodes(onnx_graph.node)
        
        # 解析输出
        self._parse_outputs(onnx_graph.output)
        
        return self.graph
    
    def _parse_initializers(self, initializers):
        """解析初始权重"""
        for init in initializers:
            tensor = self._initializer_to_tensor(init)
            self.graph.add_tensor(tensor)
    
    def _initializer_to_tensor(self, init) -> Tensor:
        """将ONNX初始化器转换为Tensor"""
        # 获取数据类型
        dtype = self._get_dtype(init.data_type)
        
        # 获取形状
        shape = list(init.dims)
        
        # 获取数据
        data = onnx.numpy_helper.to_array(init)
        
        return Tensor(
            name=init.name,
            shape=shape,
            dtype=dtype,
            data=data
        )
    
    def _parse_inputs(self, inputs):
        """解析图输入"""
        # 跳过已经在initializer中的输入
        initializer_names = set(self.graph.tensors.keys())
        
        for input_info in inputs:
            if input_info.name in initializer_names:
                continue
            
            # 获取形状
            shape = []
            if input_info.type.tensor_type.shape.dim:
                for dim in input_info.type.tensor_type.shape.dim:
                    if dim.dim_value > 0:
                        shape.append(dim.dim_value)
                    elif dim.dim_param:
                        # 动态维度，使用默认值
                        shape.append(1)
            
            # 获取数据类型
            dtype = self._get_dtype(input_info.type.tensor_type.elem_type)
            
            tensor = Tensor(
                name=input_info.name,
                shape=shape,
                dtype=dtype,
                data=None
            )
            self.graph.add_tensor(tensor)
            self.graph.inputs.append(input_info.name)
    
    def _parse_nodes(self, nodes):
        """解析计算节点"""
        for onnx_node in nodes:
            node = self._onnx_node_to_node(onnx_node)
            self.graph.add_node(node)
            
            # 为输出创建张量
            for output_name in node.outputs:
                if output_name not in self.graph.tensors:
                    # 创建占位张量
                    tensor = Tensor(
                        name=output_name,
                        shape=[],  # 形状将在后续推断
                        dtype="float32",
                        data=None
                    )
                    self.graph.add_tensor(tensor)
    
    def _onnx_node_to_node(self, onnx_node) -> Node:
        """将ONNX节点转换为Node"""
        # 解析属性
        attrs = {}
        for attr in onnx_node.attribute:
            attrs[attr.name] = self._get_attr_value(attr)
        
        return Node(
            op_type=onnx_node.op_type,
            name=onnx_node.name if onnx_node.name else f"{onnx_node.op_type}_{len(self.graph.nodes)}",
            inputs=list(onnx_node.input),
            outputs=list(onnx_node.output),
            attrs=attrs
        )
    
    def _parse_outputs(self, outputs):
        """解析图输出"""
        for output_info in outputs:
            self.graph.outputs.append(output_info.name)
    
    def _get_dtype(self, data_type: int) -> str:
        """获取数据类型字符串"""
        dtype_map = {
            1: "float32",
            2: "uint8",
            3: "int8",
            4: "uint16",
            5: "int16",
            6: "int32",
            7: "int64",
            8: "string",
            9: "bool",
            10: "float16",
            11: "float64",
            12: "uint32",
            13: "uint64",
        }
        return dtype_map.get(data_type, "float32")
    
    def _get_attr_value(self, attr) -> any:
        """获取属性值"""
        if attr.type == onnx.AttributeProto.INT:
            return attr.i
        elif attr.type == onnx.AttributeProto.INTS:
            return list(attr.ints)
        elif attr.type == onnx.AttributeProto.FLOAT:
            return attr.f
        elif attr.type == onnx.AttributeProto.FLOATS:
            return list(attr.floats)
        elif attr.type == onnx.AttributeProto.STRING:
            return attr.s.decode('utf-8')
        elif attr.type == onnx.AttributeProto.TENSOR:
            return onnx.numpy_helper.to_array(attr.t)
        else:
            return None


def parse_onnx(model_path: str) -> Graph:
    """便捷函数：解析ONNX模型"""
    parser = ONNXParser()
    return parser.parse(model_path)