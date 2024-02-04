import onnx
from onnx import shape_inference
import sys
from tabulate import tabulate
from onnx import onnx_ml_pb2 as xpb2

# sys.path.append(path.abspath('lab15-1\onnx-model-analysis-tool\src'))
# import main
# import utils
# import get_info

onnx_model = onnx.load("lenet.onnx", load_external_data=False)
onnx.checker.check_model(onnx_model)

inferred_model = shape_inference.infer_shapes(onnx_model)
print('shape inference complete ...')




def get_attribute(graph: xpb2.GraphProto):
    try:
        conv_attr = []
        for i, node in enumerate(inferred_model.graph.node):
                if node.name == "":
                    inferred_model.graph.node[i].name = str(i)
        # get the idx_list
        # idx_list = get_op_type_name_and_idx(graph)
        node_nlist = [k.name for k in graph.node]
        idx_list = {}
        for node in graph.node:
            if node.op_type in idx_list:
                idx_list[node.op_type][node.name] = node_nlist.index(node.name)
            else:
                idx_list[node.op_type] = {
                    node.name: node_nlist.index(node.name)
                }
        # traverse the idx_list['Conv']
        if 'Conv' not in idx_list.keys():
            print('[ERROR MASSAGE] This graph has no operators "Conv" !')
        else:
            for idx in idx_list['Conv'].values():
                # temp_list = utils._parse_Conv_get_pads_strides(idx, graph)
                temp_list = []
                # print('===========================================================')
                # print('[DEBUG MASSAGE]')
                # for elem in graph.node[op_idx].attribute:
                #     print(elem)

                attri_nlist = []
                # get attribute name list
                for elem in graph.node[idx].attribute:
                    attri_nlist.append(elem.name)

                # find pads
                if 'pads' in attri_nlist:
                    idx1 = attri_nlist.index('pads')
                    temp_list.append(graph.node[idx].attribute[idx1].ints)
                else:
                    temp_list.append('None')

                # strides
                if 'strides' in attri_nlist:
                    idx1 = attri_nlist.index('strides')
                    temp_list.append(graph.node[idx].attribute[idx1].ints)
                else:
                    temp_list.append('None')
                temp_tuple = (graph.node[idx].name, temp_list[0], temp_list[1])
                print(temp_tuple)




        # attr=get_info.get_Gemm_attribute(graph)

        # init Gemm_attr
        list_Gemm_attr = []

        # traverse the Gemm operators and parse the attributeProtos
        if 'Gemm' not in idx_list.keys():
            print('[ERROR MASSAGE] This graph has no operators "Gemm" !')
            return False
        else:
            # traverse the all the Gemm operator
            for idx in idx_list['Gemm'].values():
                # traverse the attributes of a Gemm operator and get the name list
                attri_nlist = [k.name for k in graph.node[idx].attribute]
                # get transA
                if 'transA' in attri_nlist:
                    attr_idx = attri_nlist.index('transA')
                    transA = graph.node[idx].attribute[attr_idx].i
                else:
                    transA = 0
                # get transB
                if 'transB' in attri_nlist:
                    attr_idx = attri_nlist.index('transB')
                    transB = graph.node[idx].attribute[attr_idx].i
                else:
                    transB = 0
                # get alpha
                if 'alpha' in attri_nlist:
                    attr_idx = attri_nlist.index('alpha')
                    alpha = graph.node[idx].attribute[attr_idx].f
                else:
                    alpha = float(1.0)
                # get beta
                if 'beta' in attri_nlist:
                    attr_idx = attri_nlist.index('beta')
                    beta = graph.node[idx].attribute[attr_idx].f
                else:
                    beta = float(1.0)
                # collect the information
                temp_tuple = (graph.node[idx].name, transA, transB, alpha, beta)
                # append to the list
                list_Gemm_attr.append(temp_tuple)

        print(list_Gemm_attr)

    except Exception as e:
        print("Unable to display: " + str(e))
        return False

    return True

#從這裡開始
print("start")
get_attribute(inferred_model.graph)



