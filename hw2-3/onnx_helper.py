"""
Onnx model helper
Author: NTHU CS 4 楊子慶
"""

from util import *
from typing import Literal, Union
import onnx
from functools import reduce
from onnx import onnx_ml_pb2, shape_inference, checker, helper

class OnnxHelper:
    NodeInfoType = Union[Literal["input"],
                         Literal["output"],
                         Literal["attribute"]]
    
    def __init__(self, model: onnx.ModelProto, to_check = False):
        """
        :param `to_check`: check model in initialization 

        Use `self.nodes` to access node information
        """
        self.model = model

        # print(f"ir_version of {self.model.graph.name} is {self.model.ir_version}")
        # print(f"opset_import is {self.model.opset_import}")
        
        self.to_check = to_check
        
        self.input_map: dict[str, int]
        self.init_map: dict[str, int]
        self.value_map: dict[str, int]
        self.output_map: dict[str, int]

        # generate basic info
        self.nodes: list[dict[str, dict]]
        self.named_object: set[str]
        self.indexing()
        if len(self.nodes) == 0:
            raise ValueError("[OnnxHelper] There is NO node exist in this model!")

    def indexing(self):
        """
        Indexing node information.
        Use `self.nodes` to access them after indexing.

        The index of `self.nodes` of the same node
            is the same as that in `self.model.graph.node`
        """
        self.model = shape_inference.infer_shapes(self.model)
        if self.to_check:
            checker.check_model(self.model, True)
        # initialize members
        graph = self.model.graph # alias
        self.input_map = { k.name: n for n, k in enumerate(graph.input) }
        self.init_map = { k.name: n for n, k in enumerate(graph.initializer) }
        self.value_map = { k.name: n for n, k in enumerate(graph.value_info) }
        self.output_map = { k.name: n for n, k in enumerate(graph.output) }
        
        self.nodes: list[dict[str, dict]] = []
        self.named_object: set[str] = set()
        # indexing
        json = []
        for n in graph.node:
            self.named_object.add(n.name)
            
            data_val: dict[OnnxHelper.NodeInfoType, Any] = {}
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
            # fetch attribute info
            attr_val = self.fetch_attr(n)
            if len(attr_val) > 0:
                data_val["attribute"] = attr_val
            json.append({ n.op_type: data_val })

        self.nodes = json
    
    def subscript(self,
                  idx: int,
                  type: Union[Literal["input"],
                              Literal["output"],
                              Literal["initializer"],
                              Literal["value_info"],
                              Literal["node"]]):
        if type == "input":
            return self.model.graph.input[idx]
        elif type == "output":
            return self.model.graph.output[idx]
        elif type == "initializer":
            return self.model.graph.initializer[idx]
        elif type == "value_info":
            return self.model.graph.value_info[idx]
        else:
            return None

    def fetch_proto(self, fid: str):
        """
        Fetch dimension and size information using field identifier `fid`
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
    
    def summary(self,
                wrap_graph_name = False,
                verbose = False,
                refresh = False,
                override_name: str = None) -> Union[dict, list]:
        """
        if `wrap_graph_name` is True, then returns a dict
            in format `{ graph_name: list_of_nodes }`
        otherwise returns `list_of_nodes`
        """
        if refresh:
            self.indexing()
        
        json = self.nodes

        if wrap_graph_name:
            json = { self.model.graph.name if override_name is None else
                     override_name : json }
            
        if verbose:
            print("\n".join(json_stringify(json)))
            
        return json
    
    def select_all(self, where: Callable[[onnx.NodeProto], bool]) -> list[int]:
        """
        Select all `node_type` nodes in model.
        returns list of node identifier to access `self.nodes`
        """
        return [cnt for cnt, n in enumerate(self.model.graph.node) if where(n)]

    def get_info_from_node(self, idx: int) -> tuple[str, dict[NodeInfoType, dict]]:
        return [*self.nodes[idx].items()][0]

    def get_named_list(self):
        return set(self.named_object)
    
    def replace_all(self, replacements: List[tuple[int, onnx_ml_pb2.GraphProto]]):
        """
        Replace all the nodes w.r.t. given node indices and re-indexing.
        replacements: list of `<< old_node_idx, new_graph >>` pair
        """
        replacements.sort(key = lambda idx: -idx[0])
        
        this_graph = self.model.graph
        for node_idx, replacement in replacements:
            popped_node = this_graph.node.pop(node_idx)
            for n in reversed(replacement.node):
                this_graph.node.insert(node_idx, n)
            this_graph.value_info.extend(replacement.value_info)
            this_graph.initializer.extend(replacement.initializer)
        
        self.to_check = True
        self.indexing()
