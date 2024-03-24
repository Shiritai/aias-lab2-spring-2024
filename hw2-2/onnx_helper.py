"""
Onnx model helper
Author: NTHU CS 4 楊子慶
"""

from util import *
from typing import Literal, Union
import onnx
from functools import reduce
from onnx import onnx_ml_pb2, shape_inference

class OnnxHelper:
    def __init__(self, model):
        self.model = shape_inference.infer_shapes(model)
        graph = self.model.graph # alias
        self.input_map = { k.name: n for n, k in enumerate(graph.input) }
        self.init_map = { k.name: n for n, k in enumerate(graph.initializer) }
        self.value_map = { k.name: n for n, k in enumerate(graph.value_info) }
        self.output_map = { k.name: n for n, k in enumerate(graph.output) }

    def fetch_proto(self, fid: str):
        """
        Fetch `ValueInfoProto` or `TensorProto` using field identifier `fid`
        """
        graph = self.model.graph # alias
        idx = self.input_map.get(fid, None)
        proto = None
        if idx is not None:
            proto = graph.input[idx]
        else:
            idx = self.init_map.get(fid, None)
            if idx is not None:
                proto = graph.initializer[idx]
            else:
                idx = self.value_map.get(fid, None)
                if idx is not None:
                    proto = graph.value_info[idx]
                else:
                    idx = self.output_map.get(fid, None)
                    if idx is not None:
                        proto = graph.output[idx]
                    else:
                        raise ValueError(f"Invalid node field identifier: {fid}")
        return proto
        
    def fetch_size(_, proto: Union[onnx.ValueInfoProto, onnx.TensorProto]) -> \
        dict[Union[Literal["shape"],
                   Literal["memory_size"]],
             int]:
        """
        Fetch shape and memory usage information w.r.t. given `proto`

        Returns dict form of `{ "shape", "memory_size" }`, \
            unit of `memory_size` is byte
        """
        
        dims = ()
        dtype = -1
        if type(proto) == onnx.ValueInfoProto:
            shape = proto.type.tensor_type.shape
            dims = tuple(map(lambda d: d.dim_value,
                             shape.dim))
            dtype = proto.type.tensor_type.elem_type
        elif type(proto) == onnx.TensorProto:
            dims = tuple(proto.dims)
            dtype = proto.data_type
        else:
            raise ValueError(f"Proto {proto} is unrecognizable")
        # filter out zero value dimension
        zero_filtered_dims = tuple(filter(lambda x: x != 0, dims))
        size = 1 if len(zero_filtered_dims) > 0 else 0
        size = reduce(lambda a, b: a * b, zero_filtered_dims, size) * _.fetch_dtype_size(dtype)
        return { "shape": dims, "memory_size": size }

    def fetch_attr(_, node: onnx_ml_pb2.NodeProto) -> dict:
        """
        Get attributes from a onnx node proto as `dict`
        """
        res = {}
        for a in node.attribute:
            val = [*a.ListFields()]
            val = val[1:-1][0][1]
            val = f"{val}".strip().split("\n")
            if len(val) > 1:
                n_dc = {}
                for v in val:
                    v = v.split(": ")
                    n_dc.update({ v[0]: v[1] })
                val = n_dc
            else:
                val = str(val[0])
            res.update({ a.name: val })
        return res

    def fetch_dtype_size(_, dtype: int):
        if dtype == onnx.TensorProto.DataType.FLOAT:
            return 4
        elif dtype == onnx.TensorProto.DataType.UINT8:
            return 1
        elif dtype == onnx.TensorProto.DataType.INT8:
            return 1
        elif dtype == onnx.TensorProto.DataType.UINT16:
            return 2
        elif dtype == onnx.TensorProto.DataType.INT16:
            return 2
        elif dtype == onnx.TensorProto.DataType.INT32:
            return 4
        elif dtype == onnx.TensorProto.DataType.INT64:
            return 8
        elif dtype == onnx.TensorProto.DataType.BOOL:
            return 1
        elif dtype == onnx.TensorProto.DataType.FLOAT16:
            return 2
        elif dtype == onnx.TensorProto.DataType.DOUBLE:
            return 8
        elif dtype == onnx.TensorProto.DataType.UINT32:
            return 4
        elif dtype == onnx.TensorProto.DataType.UINT64:
            return 8
        elif dtype == onnx.TensorProto.DataType.COMPLEX64:
            return 8
        elif dtype == onnx.TensorProto.DataType.COMPLEX128:
            return 16
        elif dtype == onnx.TensorProto.DataType.BFLOAT16:
            return 2
        else:
            raise ValueError(f"Undefined data type {dtype}")
    
    def summary(self, wrap_graph_name = False, verbose = False) -> list[str]:
        json = []
        graph = self.model.graph
        for n in graph.node:
            data_val = {}
            # fetch input info if exist
            if len(n.input) > 0:
                input_info = {}
                for i_node in n.input:
                    try:
                        p = self.fetch_proto(i_node)
                        sz = self.fetch_size(p)
                        input_info[p.name] = sz
                    except ValueError as e:
                        print(e)
                data_val["input"] = input_info
            # fetch output info if exist
            if len(n.output) > 0:
                output_info = {}
                for o_node in n.output:
                    try:
                        p = self.fetch_proto(o_node)
                        sz = self.fetch_size(p)
                        output_info[p.name] = sz
                    except ValueError as e:
                        print(e)
                data_val["output"] = output_info
            # fetch attribute
            attr_val = self.fetch_attr(n)
            if len(attr_val) > 0:
                data_val["attribute"] = attr_val
            json.append({ n.op_type: data_val })

        if wrap_graph_name:
            json = { self.model.graph.name: json }

        if verbose:
            print("\n".join(json_stringify(json)))
            
        return json

