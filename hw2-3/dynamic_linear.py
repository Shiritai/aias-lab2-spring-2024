"""
Dynamic Linear graph for shrinking size of linear, fully parameterized
Author: NTHU CS 楊子慶
"""

from util import *
import numpy as np
import onnx
from onnx import TensorProto, helper, checker
from onnx.reference import ReferenceEvaluator

class GemmInputInfo:
    def __init__(self, name: str, trans: bool = False):
        self.name = name
        self.trans = 1 if trans else 0

class GemmOutputInfo:
    def __init__(self, name: str):
        self.name = name

class DynamicLinear:
    TEST_N = 10
    
    def get_factor(side: int, max_side: int) -> tuple[int, bool]:
        """
        helper function to get splitting factor for some dimension
        return (factor, integer-divisible)
        """
        factor = side // max_side
        complement = 0 if side % max_side == 0 else 1

        return (factor + complement if factor != 0 else 1, complement == 0)

    def __init__(self,
                 m: int,
                 k: int,
                 n: int,
                 m_mx: int,
                 k_mx: int,
                 n_mx: int,
                 a_info:    GemmInputInfo = None,
                 b_info:    GemmInputInfo = None,
                 bias_info: GemmInputInfo = None,
                 c_info:    GemmOutputInfo = None,
                 name: str = "dl",
                 named_ls: set[str] = None
                 ):
        """
        Graph/model equivalent to Linear (`y = ax + b`) operator (i.e. gemm)
            with customizable multiplication space complexity
        
        `m`, `k`, `n`: are dimension of two matrix to be multiplied
        `m_mx`, `k_mx`, `n_mx`: determine splitting factors of each dimension
        `DynamicLinear` will make sure that space complexity of Matrix Multiplication is

        :math:
            O(m_mx \\times n_mx \\times k_mx)

        Usage:
        
        ```python
        dl_model = DynamicLinear(m, k, n, m_mx, n_mx, k_mx)
        
        dl_model.test_model
        dl_model.test_graph # equivalent to dl_model.test_model.graph

        dl_model.model
        dl_model.graph # equivalent to dl_model.model.graph
        ```
        """
        self.m, self.k, self.n, self.m_mx, self.k_mx, self.n_mx = m, k, n, m_mx, k_mx, n_mx

        self.a_info, self.b_info, self.bias_info, self.c_info = a_info, b_info, bias_info, c_info

        self.name = name if name == "dl" else f"dl_{name}"
        self.named_ls = named_ls if named_ls is not None else set()
        self.io_named_ls = set()
        self.test_n = DynamicLinear.TEST_N
        
        self.initialize_test_graph()
        self.initialize_graph()
        
    def name_it(self, name: str, is_io = False) -> str:
        """
        Name a new object and manage named object list to avoid name conflict
        """
        if is_io:
            # For i/o case, there is no need to record i/o in named_ls
            # 
            # Note that our test model only has i/o nodes,
            #   so there will be no side effect if we build
            #   test model before our dynamic linear graph.
            # Instead, we record them in self.named_list for
            #   generating named list of our graph.
            # 
            # If self.named_ls in inherited from other model,
            #   which i/o also delivered to this graph,
            #   then no error will occur as they're not count as conflict.
            self.io_named_ls.add(name)
            return name

        res = f"{self.name}_{name}"
        while res in self.named_ls:
            res += "_"
        self.named_ls.add(res)
        return res

    def get_io_info(self) -> tuple[tuple[str, str, str], str]:
        if self.a_info is not None:
            a_name, trans_a = self.a_info.name, self.a_info.trans
            a_shape = (self.m, self.k) if not self.a_info.trans else (self.k, self.m)
        else:
            a_name, trans_a = "A", False
            a_shape = (self.m, self.k)

        if self.b_info is not None:
            b_name, trans_b = self.b_info.name, self.b_info.trans
            b_shape = (self.k, self.n) if not self.b_info.trans else (self.n, self.k)
        else:
            b_name, trans_b = "B", False
            b_shape = (self.k, self.n)
            
        if self.bias_info is not None:
            bias_name = self.bias_info.name
        else:
            bias_name = "Bias"
            
        if self.c_info is not None:
            c_name = self.c_info.name
        else:
            c_name = "C"

        return ((a_name, trans_a, a_shape),
                (b_name, trans_b, b_shape), bias_name), c_name
        
    def initialize_test_graph(self):
        """
        Initialize graph and model equivalent to those of DynamicLinear for testing
        """
        ((a_name, trans_a, a_shape),
         (b_name, trans_b, b_shape), bias_name), c_name = self.get_io_info()

        a_name = self.name_it(a_name, True)
        b_name = self.name_it(b_name, True)
        bias_name = self.name_it(bias_name, True)
        c_name = self.name_it(c_name, True)
        
        self.test_graph = helper.make_graph(nodes=[helper.make_node("Gemm",
                                                                    inputs=(a_name, b_name, bias_name),
                                                                    outputs=(c_name,),
                                                                    transA=trans_a,
                                                                    transB=trans_b,
                                                                    ),],
                                            name=f"TestGraph_{self.name}",
                                            inputs=[helper.make_tensor_value_info(a_name,
                                                                                  TensorProto.FLOAT,
                                                                                  a_shape),
                                                    helper.make_tensor_value_info(b_name,
                                                                                  TensorProto.FLOAT,
                                                                                  b_shape),
                                                    helper.make_tensor_value_info(bias_name,
                                                                                  TensorProto.FLOAT,
                                                                                  (self.m, self.n))],
                                            outputs=[helper.make_tensor_value_info(c_name,
                                                                                   TensorProto.FLOAT,
                                                                                   (self.m, self.n))],)
        self.test_model = helper.make_model(self.test_graph)

    def initialize_graph(self):
        """
        Create a Linear graph with generalization of all dimension factors
        
        The implementation simulates the code snippet below,
        i.e. Matrix multiplication
        
        ```python
        c_on_m_axis = []
        for _m in range(m_factor):
            c_on_n_axis = [0] * n_factor
            for _n in range(n_factor):
                mm_list = []
                for _k in range(k_factor):
                    mm_list.append(a_splits[_k][_m] * b_splits[_n][_k])
                c_on_n_axis[_n] = sum(mm_list)
            c_list = concat(c_on_n_axis)
        c = concat(c_on_m_axis)
        ```
        """
        m, k, n, m_mx, n_mx, k_mx = \
            self.m, self.k, self.n, \
                self.m_mx, self.n_mx, self.k_mx
        # define factors
        m_factor, m_is_div = DynamicLinear.get_factor(m, m_mx)
        k_factor, k_is_div = DynamicLinear.get_factor(k, k_mx)
        n_factor, n_is_div = DynamicLinear.get_factor(n, n_mx)
        
        m_slices = self.name_it("m_slices")
        m_slice_list = [m_mx if m_is_div or m_cnt + 1 != m_factor
                             else m % m_mx for m_cnt in range(m_factor)]
        m_slice_tensor = helper.make_tensor(m_slices,
                                            TensorProto.INT64,
                                            (m_factor, ),
                                            m_slice_list)

        k_slices = self.name_it("k_slices")
        k_slice_list = [k_mx if k_is_div or k_cnt + 1 != k_factor
                             else k % k_mx for k_cnt in range(k_factor)]
        k_slice_tensor = helper.make_tensor(k_slices,
                                            TensorProto.INT64,
                                            (k_factor, ),
                                            k_slice_list)

        n_slices = self.name_it("n_slices")
        n_slice_list = [n_mx if n_is_div or n_cnt + 1 != n_factor
                             else n % n_mx for n_cnt in range(n_factor)]
                             
        n_slice_tensor = helper.make_tensor(n_slices,
                                            TensorProto.INT64,
                                            (n_factor, ),
                                            n_slice_list)
        
        if self.a_info is not None:
            a, a_pre_input = self.a_info.name, self.a_info.name
            if self.a_info.trans:
                a = a_pre_input + "_real"
                a_pre_input = self.name_it(a_pre_input, True)
                a = self.name_it(a)
        else:
            a, a_pre_input = self.name_it("A", True), ""
            
        a_node = helper.make_tensor_value_info(a,
                                               TensorProto.FLOAT,
                                               (m, k))
        
        if self.b_info is not None:
            b, b_pre_input = self.b_info.name, self.b_info.name
            if self.b_info.trans:
                b = b_pre_input + "_real"
                b_pre_input = self.name_it(b_pre_input, True)
                b = self.name_it(b)
        else:
            b, b_pre_input = self.name_it("B", True), ""

        b_node = helper.make_tensor_value_info(b,
                                               TensorProto.FLOAT,
                                               (k, n))
        
        if self.bias_info is not None:
            bias = self.bias_info.name
        else:
            bias = "Bias"
        bias = self.name_it(bias, True)
        
        bias_node = helper.make_tensor_value_info(bias,
                                                  TensorProto.FLOAT,
                                                  (m, n))
    
        # TODO: use Reshape
        # # Reshape: equivalent to first-and-second-level for loops in matrix multiplication
        # #   First, define inner output tensor of reshape result
        # a_reshapes = [[f"A_Reshape_{_m}-{_k}" for _m in range(m_factor)] for _k in range(k_factor)]
        # b_reshapes = [[f"B_Reshape_{_k}-{_n}" for _k in range(k_factor)] for _n in range(n_factor)]
        # #   Second define new shape
        # a_new_shape = (m_factor * k_factor, m_mx, k_mx)
        # b_new_shape = (k_factor * n_factor, k_mx, n_mx)
        
        # a_new_shape_tensor = helper.make_tensor_value_info("a_new_shape",
        #                                                    TensorProto.INT64,
        #                                                    a_new_shape)
        # b_new_shape_tensor = helper.make_tensor_value_info("b_new_shape",
        #                                                    TensorProto.INT64,
        #                                                    b_new_shape)
        # #   Third, define tensor info w.r.t. inner output tensor of reshape result
        # a_reshape_tensors = [[helper.make_tensor_value_info(a_reshapes[_k][_m],
        #                                                    TensorProto.FLOAT,
        #                                                    a_new_shape) for _m in range(m_factor)] for _k in range(k_factor)]
        # b_reshape_tensors = [[helper.make_tensor_value_info(b_reshapes[_n][_k],
        #                                                    TensorProto.FLOAT,
        #                                                    b_new_shape) for _k in range(k_factor)] for _n in range(n_factor)]
        # #   Preparation complete, create Reshape layers
        # a_reshape_layer = helper.make_node("Reshape",
        #                                    inputs=[a, "a_new_shape"],
        #                                    outputs=flatten(a_reshapes),
        #                                    name="ReshapeA")
        # b_reshape_layer = helper.make_node("Reshape",
        #                                    inputs=[b, "b_new_shape"],
        #                                    outputs=flatten(b_reshapes),
        #                                    name="ReshapeB")
        
        # Split (1st): equivalent to first-level for loops in matrix multiplication
        #   First, define inner output tensor of split result
        a_m_splits = [self.name_it(f"A_Split_m{_m}") for _m in range(m_factor)]
        b_k_splits = [self.name_it(f"B_Split_k{_k}") for _k in range(k_factor)]
        #   Second, define tensor info w.r.t. inner output tensor of split result
        a_m_split_tensors = [helper.make_tensor_value_info(a_m_split,
                                                           TensorProto.FLOAT,
                                                           (m_cnt, k))
                             for m_cnt, a_m_split in zip(m_slice_list, a_m_splits)]
        b_k_split_tensors = [helper.make_tensor_value_info(b_k_split,
                                                           TensorProto.FLOAT,
                                                           (k_cnt, n))
                             for k_cnt, b_k_split in zip(k_slice_list, b_k_splits)]
        #   Preparation complete, create Split layers
        a_m_split_layer = helper.make_node("Split",
                                           inputs=[a, m_slices],
                                           outputs=a_m_splits,
                                           name=self.name_it("SplitA_m"),
                                           axis=0)
        b_k_split_layer = helper.make_node("Split",
                                           inputs=[b, k_slices],
                                           outputs=b_k_splits,
                                           name=self.name_it("SplitB_k"),
                                           axis=0)
        self.check_node([a_m_split_layer])
        self.check_node([b_k_split_layer])
        
        # Split (2nd): equivalent to second-level for loops in matrix multiplication
        #   First, define inner output tensor of split result
        a_splits = [[self.name_it(f"A_Split_m{_m}_k{_k}")
                     for _m in range(m_factor)]
                    for _k in range(k_factor)]
        b_splits = [[self.name_it(f"B_Split_k{_k}_n{_n}")
                     for _k in range(k_factor)]
                    for _n in range(n_factor)]
        #   Second, define tensor info w.r.t. inner output tensor of split result
        a_split_tensors = [[helper.make_tensor_value_info(a_splits[_k][_m],
                                                         TensorProto.FLOAT,
                                                         (m_cnt, k_cnt))
                            for _m, m_cnt in enumerate(m_slice_list)]
                           for _k, k_cnt in enumerate(k_slice_list)]
        b_split_tensors = [[helper.make_tensor_value_info(b_splits[_n][_k],
                                                         TensorProto.FLOAT,
                                                         (k_cnt, n_cnt))
                            for _k, k_cnt in enumerate(k_slice_list)]
                           for _n, n_cnt in enumerate(n_slice_list)]
        #   Preparation complete, create Split layers
        a_split_layers = [helper.make_node("Split",
                                           inputs=[a_m_splits[_m], k_slices],
                                           outputs=[a_splits[_k][_m] for _k in range(k_factor)],
                                           name=self.name_it(f"SplitA_m{_m}"),
                                           axis=1)
                          for _m in range(m_factor)]
        b_split_layers = [helper.make_node("Split",
                                           inputs=[b_k_splits[_k], n_slices],
                                           outputs=[b_splits[_n][_k] for _n in range(n_factor)],
                                           name=self.name_it(f"SplitB_k{_k}"),
                                           axis=1)
                          for _k in range(k_factor)]
        self.check_node(a_split_layers)
        self.check_node(b_split_layers)

        # MatMul: equivalent to k-level for loop of matrix multiplication
        #   a_splits and b_splits are the input of this layer
        #   
        #   First define name of inner output with radix (m, n, k)
        #       notice that radix like (m, n, k) should change its mark like a counter
        mms = [[[self.name_it(f"MatMul_m{_m}_n{_n}_k{_k}")
                 for _m in range(m_factor)]
                for _n in range(n_factor)]
               for _k in range(k_factor)]
        #   Second, define inner output of general matrix multiplication result
        mm_tensors = [[[helper.make_tensor_value_info(mms[_k][_n][_m],
                                                      TensorProto.FLOAT,
                                                      (m_cnt, n_cnt))
                        for _m, m_cnt in enumerate(m_slice_list)] 
                       for _n, n_cnt in enumerate(n_slice_list)]
                      for _k in range(k_factor)]
        #   Preparation complete, create MatMul layers
        mm_layers = [[[helper.make_node("MatMul",
                                        inputs=[a_splits[_k][_m], b_splits[_n][_k]],
                                        # inputs=[a_reshapes[_k][_m], b_reshapes[_n][_k]], # TODO use Reshape
                                        outputs=[mms[_k][_n][_m]],
                                        name=mms[_k][_n][_m])
                       for _m in range(m_factor)]
                      for _n in range(n_factor)]
                     for _k in range(k_factor)]
        
        # Sum: equivalent to sum all middle result of multiplication in k-level loop
        #   mms are the input of this layer
        #
        #   First, define name of inner output with radix ONLY (m, n)
        #       since we'll eliminate k-level after summation
        sums = [[self.name_it(f"Sum_m{_m}_n{_n}")
                 for _m in range(m_factor)]
                for _n in range(n_factor)]
        #   Second, define corresponding value info
        sum_tensors = [[helper.make_tensor_value_info(sums[_n][_m],
                                                      TensorProto.FLOAT,
                                                      (m_cnt, n_cnt))
                        for _m, m_cnt in enumerate(m_slice_list)]
                       for _n, n_cnt in enumerate(n_slice_list)]
        #   Preparation complete, create Sum layers
        sum_layers = [[helper.make_node("Sum",
                                        inputs=[mms[_k][_n][_m] for _k in range(k_factor)],
                                        outputs=[sums[_n][_m]],
                                        name=sums[_n][_m])
                       for _m in range(m_factor)]
                      for _n in range(n_factor)]

        
        # Concat (n level): equivalent to finish n-level for loop
        #   sums are the input of this layer
        #
        #   First, define name of inner output with radix ONLY m
        #       since we'll eliminate n-level after concatenating
        concat_ns = [self.name_it(f"Concat_m{_m}") for _m in range(m_factor)]
        #   Second, define corresponding value info
        #       Notice that there will be m_factor s of tensors of shape (m_mx, n)
        #           so they'll eventually be combined to a (m, n) matrix
        concat_n_tensors = [helper.make_tensor_value_info(concat_ns[_m],
                                                          TensorProto.FLOAT,
                                                          (m_cnt, n))
                            for _m, m_cnt in enumerate(m_slice_list)]
        #   Preparation complete, create Concat (n level) layers
        concat_n_layers = [helper.make_node("Concat",
                                            inputs=[sums[_n][_m] for _n in range(n_factor)],
                                            outputs=[concat_ns[_m]],
                                            name=concat_ns[_m],
                                            axis=1)
                           for _m in range(m_factor)]
        self.check_node(concat_n_layers)
        
        # Concat (m level): equivalent to finish n-level for loop
        #   concat_ns are the input of this layer
        #
        #   First, define name of c
        concat_m = self.name_it("Concat_M")
        concat_m_tensor = helper.make_tensor_value_info(concat_m,
                                               TensorProto.FLOAT,
                                               (m, n))
        #   Preparation complete, create Concat (m level) layers
        concat_m_layer = helper.make_node("Concat",
                                          inputs=[concat_ns[_m] for _m in range(m_factor)],
                                          outputs=[concat_m],
                                          name=concat_m,
                                          axis=0)
        
        # Add: addition operator to implement y = ax + bias
        #   concat_m is the input of this layer
        #
        #   First, define name of output, i.e. concat_m_tensor
        #       since we're concatenating
        output = self.name_it(self.c_info.name if self.c_info is not None else "C", True)
        output_node = helper.make_tensor_value_info(output,
                                                    TensorProto.FLOAT,
                                                    (m, n))
        #   Preparation complete, create Add layers
        addition_layer = helper.make_node("Add",
                                          inputs=[concat_m, bias],
                                          outputs=[output],
                                          name=output)

        self.inputs = [a_node, b_node, bias_node]

        self.outputs = [output_node]

        self.nodes = [# a_reshape_layer, # TODO: use Reshape
                 # b_reshape_layer, # TODO: use Reshape
                 a_m_split_layer,
                 b_k_split_layer,
                 *a_split_layers,
                 *b_split_layers,
                 *flatten(mm_layers),
                 *flatten(sum_layers),
                 *flatten(concat_n_layers),
                 concat_m_layer,
                 addition_layer]
        
        self.value_info = [# a_new_shape_tensor, # TODO: use Reshape
                      # b_new_shape_tensor, # TODO: use Reshape
                      # *flatten(a_reshape_tensors), # TODO use Reshape
                      # *flatten(b_reshape_tensors), # TODO use Reshape
                      *flatten(a_m_split_tensors),
                      *flatten(b_k_split_tensors),
                      *flatten(a_split_tensors),
                      *flatten(b_split_tensors),
                      *flatten(mm_tensors),
                      *flatten(sum_tensors),
                      *flatten(concat_n_tensors),
                      concat_m_tensor]

        # define transpose layer of needed
        if self.a_info is not None and self.a_info.trans:
            # input of transpose layer
            a_pre_input_node = helper.make_tensor_value_info(a_pre_input,
                                                             TensorProto.FLOAT,
                                                             (k, m))
            a_trans_layer = helper.make_node("Transpose",
                                             inputs=[a_pre_input,],
                                             outputs=[a],
                                             name=self.name_it("TransposeA"))
            # change input A to that before transposition
            self.inputs[0] = a_pre_input_node
            # add transpose layer and a (inner output) to value_info
            # Note: order of layers matters, onnx will not do
            #   topological sort before inferring
            self.nodes.insert(0, a_trans_layer)
            self.value_info.append(a_node)
            
        if self.b_info is not None and self.b_info.trans:
            # input of transpose layer
            b_pre_input_node = helper.make_tensor_value_info(b_pre_input,
                                                             TensorProto.FLOAT,
                                                             (n, k))
            b_trans_layer = helper.make_node("Transpose",
                                             inputs=[b_pre_input,],
                                             outputs=[b],
                                             name=self.name_it("TransposeB"))
            # change input B to that before transposition
            self.inputs[1] = b_pre_input_node
            # add transpose layer and b (inner output) to value_info
            # Note: order of layers matters, onnx will not do
            #   topological sort before inferring
            self.nodes.insert(0, b_trans_layer)
            self.value_info.append(b_node)
    
        self.initializer = [m_slice_tensor,
                            k_slice_tensor,
                            n_slice_tensor,]
        # self.initializer = []

        # check model status
        for n in [*self.inputs, *self.outputs, *self.value_info]:
            checker.check_value_info(n)
            
        self.check_node(self.nodes)

        for n in self.initializer:
            checker.check_tensor(n)


        # All parameters/activations prepared, build model
        self.make_model()
    
    def get_name_list(self):
        return set(*self.named_ls, *self.io_named_ls)

    def make_model(self):
        self.make_graph()
        self.model = helper.make_model(self.graph)
        checker.check_model(self.model, True)
        
    def make_graph(self):
        self.graph = helper.make_graph(nodes=self.nodes,
                                       name=f"DynamicLinear_{self.name}",
                                       inputs=self.inputs,
                                       outputs=self.outputs,
                                       value_info=self.value_info,
                                       initializer=self.initializer)
        checker.check_graph(self.graph)
        # print(self.graph.input)
        
    def check_node(self, nodes: list[onnx.NodeProto]):
        for n in nodes:
            checker.check_node(n)
        
    def test_equivalence(self) -> int:
        """
        Check whether model generated is equivalent to testing model
            using random input test
        
        returns cases we've run
        """
        session_test = ReferenceEvaluator(self.test_model)
        session_dl = ReferenceEvaluator(self.model)

        def random_test():
            # A, B, bias
            ((a_name, _, a_shape),
             (b_name, _, b_shape), bias_name), _ = self.get_io_info()

            A = np.random.randn(*a_shape).astype(np.float64)
            B = np.random.randn(*b_shape).astype(np.float64)
            bias = np.random.randn(self.m, self.n).astype(np.float64)

            feeds = { a_name: A, b_name: B, bias_name: bias }
            
            result_t = session_test.run(None, feeds)
            result_d = session_dl.run(None, feeds)

            assert np.allclose(result_t, result_d), \
                   f"({self.__class__}) [Error] Can't pass unit test :("
        
        for _ in range(self.test_n):
            random_test()
        
        return self.test_n
        
