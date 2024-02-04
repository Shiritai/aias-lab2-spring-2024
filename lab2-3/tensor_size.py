import onnx
from onnx import shape_inference
from os import path
import sys
from tabulate import tabulate
from onnx import onnx_ml_pb2 as xpb2

onnx_model = onnx.load("lenet.onnx", load_external_data=False)
onnx.checker.check_model(onnx_model)

inferred_model = shape_inference.infer_shapes(onnx_model)
print('shape inference complete ...')


def _parse_element(elem: xpb2.ValueInfoProto):
    name = getattr(elem, 'name', "None")
    data_type = "NA"
    shape_str = "NA"
    etype = getattr(elem, 'type', False)
    if etype:
        ttype = getattr(etype, 'tensor_type', False)
        if ttype:
            data_type = getattr(ttype, 'elem_type', 0)
            shape = getattr(elem.type.tensor_type, "shape", False)
            if shape:
                shape_str = "["
                dims = getattr(shape, 'dim', [])
                for dim in dims:
                    vals = getattr(dim, 'dim_value', "?")
                    shape_str += (str(vals) + ",")
                shape_str = shape_str.rstrip(",")
                shape_str += "]"
    return name, data_type, shape_str

def list_tensor_size(graph: xpb2.GraphProto):
    # main.List_Tensor_SizeOfConv(graph)
    try:
        for i, node in enumerate(inferred_model.graph.node):
                if node.name == "":
                    inferred_model.graph.node[i].name = str(i)
        # get the list
        All_Conv_tensor_size = []
        bias = False
        # get the idx of the operators
        if type(graph) is not xpb2.GraphProto:
            sys.exit('The input graph is not a GraphProto!')

        node_nList = [k.name for k in graph.node]
        input_nlist = [k.name for k in graph.input]
        initializer_nlist = [k.name for k in graph.initializer]
        value_info_nlist = [k.name for k in graph.value_info]
        output_nlist = [k.name for k in graph.output]
        idx_list = {}
        for node in graph.node:
            if node.op_type in idx_list:
                idx_list[node.op_type][node.name] = node_nList.index(node.name)
            else:
                idx_list[node.op_type] = {
                    node.name: node_nList.index(node.name)
                }

        # get the Conv tensor size
        if 'Conv' not in idx_list.keys():
            print('This graph has no operators "Conv" !')
        else:
            for idx in idx_list['Conv'].values():
                # temp_tuple, bias = utils._Cal_tensor_size_ConvOrGemm(idx, graph)
                num_conv_input_tensor = len(graph.node[idx].input)
                list_of_data_num = []
                # get input tensor proto
                for input_name in graph.node[idx].input:
                    # get tensor data
                    if input_name in input_nlist:
                        name_idx = input_nlist.index(input_name)
                        data = graph.input[name_idx]
                        type_num = int(1)
                    elif input_name in value_info_nlist:
                        name_idx = value_info_nlist.index(input_name)
                        data = graph.value_info[name_idx]
                        type_num = int(2)
                    elif input_name in initializer_nlist:
                        name_idx = initializer_nlist.index(input_name)
                        data = graph.initializer[name_idx]
                        type_num = int(3)
                    elif input_name in output_nlist:
                        name_idx = output_nlist.index(input_name)
                        data = graph.output[name_idx]
                        type_num = int(4)
                    else:
                        print("Can't find the tensor: ", input_name)
                        print('input_nlist:\n', input_nlist)
                        print('===================')
                        print('value_info_nlist:\n', value_info_nlist)
                        print('===================')
                        print('initializer_nlist:\n', initializer_nlist)
                        print('===================')
                        print('output_nlist:\n', output_nlist)
                        print('===================')
                
                    list_of_data_num.append((data, type_num))

                if graph.node[idx].output[0] in input_nlist:
                    name_idx = input_nlist.index(graph.node[idx].output[0])
                    data = graph.input[name_idx]
                    type_num = int(1)
                elif graph.node[idx].output[0] in value_info_nlist:
                    name_idx = value_info_nlist.index(graph.node[idx].output[0])
                    data = graph.value_info[name_idx]
                    type_num = int(2)
                elif graph.node[idx].output[0] in initializer_nlist:
                    name_idx = initializer_nlist.index(graph.node[idx].output[0])
                    data = graph.initializer[name_idx]
                    type_num = int(3)
                elif graph.node[idx].output[0] in output_nlist:
                    name_idx = output_nlist.index(graph.node[idx].output[0])
                    data = graph.output[name_idx]
                    type_num = int(4)
                else:
                    print("Can't find the tensor: ", graph.node[idx].output[0])
                    print('input_nlist:\n', input_nlist)
                    print('===================')
                    print('value_info_nlist:\n', value_info_nlist)
                    print('===================')
                    print('initializer_nlist:\n', initializer_nlist)
                    print('===================')
                    print('output_nlist:\n', output_nlist)
                    print('===================')
                list_of_data_num.append((data, type_num))

                list_temp = [
                    graph.node[idx].name,
                ]
                for elem in list_of_data_num:
                    if elem[0]:
                        if elem[1] == 3:
                            name = getattr(elem[0], 'name', "None")
                            # get the data type of the tensor
                            data_type = getattr(elem[0], 'data_type', False)
                            # get the shape of the tensor
                            shape = getattr(elem[0], 'dims', [])
                        else:
                            # name, data_type, shape = utils._parse_ValueInfoProto(elem[0])
                            name, data_type, shape_str = _parse_element(elem[0])
                            shape_str = shape_str.strip('[]')
                            shape_str = shape_str.split(',')
                            shape = []
                            for dim in shape_str:
                                shape.append(int(dim))
                        mem_size = int(1)
                        # traverse the list to get the number of the elements
                        for num in shape:
                            mem_size *= num
                        # multiple the size of variable with the number of the elements
                        # "FLOAT": 1,
                        # "UINT8": 2,
                        # "INT8": 3,
                        # "UINT16": 4,
                        # "INT16": 5,
                        # "INT32": 6,
                        # "INT64": 7,
                        # # "STRING" : 8,
                        # "BOOL": 9,
                        # "FLOAT16": 10,
                        # "DOUBLE": 11,
                        # "UINT32": 12,
                        # "UINT64": 13,
                        # "COMPLEX64": 14,
                        # "COMPLEX128": 15
                        if data_type == 1:
                            mem_size *= 4
                        elif data_type == 2:
                            mem_size *= 1
                        elif data_type == 3:
                            mem_size *= 1
                        elif data_type == 4:
                            mem_size *= 2
                        elif data_type == 5:
                            mem_size *= 2
                        elif data_type == 6:
                            mem_size *= 4
                        elif data_type == 7:
                            mem_size *= 8
                        elif data_type == 9:
                            mem_size *= 1
                        elif data_type == 10:
                            mem_size *= 2
                        elif data_type == 11:
                            mem_size *= 8
                        elif data_type == 12:
                            mem_size *= 4
                        elif data_type == 13:
                            mem_size *= 8
                        elif data_type == 14:
                            mem_size *= 8
                        elif data_type == 15:
                            mem_size *= 16
                        list_temp.append(mem_size)
                    else:
                        print(graph.node[idx].name, 'tenosr no found ! Something wrong')

                if len(list_of_data_num) > 3:  # the conv has bias
                    ConvOrGemm_tensor_size = (list_temp[0], list_temp[1], list_temp[2],
                                            list_temp[3], list_temp[4], list_temp[1] +
                                            list_temp[2] + list_temp[3] + list_temp[4])
                    bias = True
                else:  # the conv has no bias
                    ConvOrGemm_tensor_size = (list_temp[0], list_temp[1], list_temp[2],
                                            list_temp[3],
                                            list_temp[1] + list_temp[2] + list_temp[3])
                    bias = False

                All_Conv_tensor_size.append(ConvOrGemm_tensor_size)
    # main.List_Tensor_SizeOfGemm(graph)
        if 'Gemm' not in idx_list.keys():
            print('This graph has no operators "Gemm" !')
        else:
            for idx in idx_list['Gemm'].values():
                num_conv_input_tensor = len(graph.node[idx].input)
                list_of_data_num = []
                for input_name in graph.node[idx].input:
                    if input_name in input_nlist:
                        name_idx = input_nlist.index(input_name)
                        data = graph.input[name_idx]
                        type_num = int(1)
                    elif input_name in value_info_nlist:
                        name_idx = value_info_nlist.index(input_name)
                        data = graph.value_info[name_idx]
                        type_num = int(2)
                    elif input_name in initializer_nlist:
                        name_idx = initializer_nlist.index(input_name)
                        data = graph.initializer[name_idx]
                        type_num = int(3)
                    elif input_name in output_nlist:
                        name_idx = output_nlist.index(input_name)
                        data = graph.output[name_idx]
                        type_num = int(4)
                    else:
                        print("Can't find the tensor: ", input_name)
                        print('input_nlist:\n', input_nlist)
                        print('===================')
                        print('value_info_nlist:\n', value_info_nlist)
                        print('===================')
                        print('initializer_nlist:\n', initializer_nlist)
                        print('===================')
                        print('output_nlist:\n', output_nlist)
                        print('===================')
                
                    list_of_data_num.append((data, type_num))
                if graph.node[idx].output[0] in input_nlist:
                    name_idx = input_nlist.index(graph.node[idx].output[0])
                    data = graph.input[name_idx]
                    type_num = int(1)
                elif graph.node[idx].output[0] in value_info_nlist:
                    name_idx = value_info_nlist.index(graph.node[idx].output[0])
                    data = graph.value_info[name_idx]
                    type_num = int(2)
                elif graph.node[idx].output[0] in initializer_nlist:
                    name_idx = initializer_nlist.index(graph.node[idx].output[0])
                    data = graph.initializer[name_idx]
                    type_num = int(3)
                elif graph.node[idx].output[0] in output_nlist:
                    name_idx = output_nlist.index(graph.node[idx].output[0])
                    data = graph.output[name_idx]
                    type_num = int(4)
                else:
                    print("Can't find the tensor: ", graph.node[idx].output[0])
                    print('input_nlist:\n', input_nlist)
                    print('===================')
                    print('value_info_nlist:\n', value_info_nlist)
                    print('===================')
                    print('initializer_nlist:\n', initializer_nlist)
                    print('===================')
                    print('output_nlist:\n', output_nlist)
                    print('===================')
                list_of_data_num.append((data, type_num))
                list_temp = [
                    graph.node[idx].name,
                ]
                for elem in list_of_data_num:
                    if elem[0]:
                        if elem[1] == 3:
                            name = getattr(elem[0], 'name', "None")
                            data_type = getattr(elem[0], 'data_type', False)
                            shape = getattr(elem[0], 'dims', [])
                        else:
                            name, data_type, shape_str = _parse_element(elem[0])
                            shape_str = shape_str.strip('[]')
                            shape_str = shape_str.split(',')
                            shape = []
                            for dim in shape_str:
                                shape.append(int(dim))
                        mem_size = int(1)
                        # traverse the list to get the number of the elements
                        for num in shape:
                            mem_size *= num
                        # multiple the size of variable with the number of the elements
                        if data_type == 'FLOAT':
                            mem_size *= 8
                        elif data_type == 'UINT8':
                            mem_size *= 1
                        elif data_type == 'INT8':
                            mem_size *= 1
                        elif data_type == 'UINT16':
                            mem_size *= 2
                        elif data_type == 'INT16':
                            mem_size *= 2
                        elif data_type == 'INT32':
                            mem_size *= 4
                        elif data_type == 'INT64':
                            mem_size *= 8
                        elif data_type == 'BOOL':
                            mem_size *= 1
                        elif data_type == 'FLOAT16':
                            mem_size *= 2
                        elif data_type == 'DOUBLE':
                            mem_size *= 8
                        elif data_type == 'UINT32':
                            mem_size *= 4
                        elif data_type == 'UINT64':
                            mem_size *= 8
                        elif data_type == 'COMPLEX64':
                            mem_size *= 8
                        elif data_type == 'COMPLEX128':
                            mem_size *= 16
                        else:
                            print("Undefined data type")
                        list_temp.append(mem_size)
                    else:
                        print(graph.node[idx].name, 'tenosr no found ! Something wrong')

                if len(list_of_data_num) > 3:  # the conv has bias
                    ConvOrGemm_tensor_size = (list_temp[0], list_temp[1], list_temp[2],
                                            list_temp[3], list_temp[4], list_temp[1] +
                                            list_temp[2] + list_temp[3] + list_temp[4])
                    bias = True
                else:  # the conv has no bias
                    ConvOrGemm_tensor_size = (list_temp[0], list_temp[1], list_temp[2],
                                            list_temp[3],
                                            list_temp[1] + list_temp[2] + list_temp[3])
                    bias = False

                All_Gemm_tensor_size.append(ConvOrGemm_tensor_size)  
        # print
        if 'Conv' not in idx_list.keys():
            print('returning from List_Tensor_SizeOfConv() ...')
        else:
            if bias:
                columns = [
                    'Conv_name', 'Input_tesnor', 'Weight_tensor',
                    'Bias_tensor', 'Output_tensor', 'Total'
                ]
            else:
                columns = [
                    'Conv_name', 'Input_tesnor', 'Weight_tensor',
                    'Output_tensor', 'Total'
                ]

            # resort the list
            All_Conv_tensor_size = sorted(All_Conv_tensor_size,
                                          key=lambda Layer: Layer[-1],
                                          reverse=True)
            print('list_of_layer RESORTED ! ')
            print(tabulate(All_Conv_tensor_size, headers=columns))
            print(
                '====================================================================================\n'
            )

        if 'Gemm' not in idx_list.keys():
            print('returning from List_Tensor_SizeOfGemm() ...')
        else:
            if bias:
                columns = [
                    'Gemm_name', 'Mat_A', 'Mat_B', 'Mat_C', 'Mat_Y', 'Total'
                ]
            else:
                columns = ['Gemm_name', 'Mat_A', 'Mat_B', 'Mat_Y', 'Total']

            # resort the list
            All_Gemm_tensor_size = sorted(All_Gemm_tensor_size,
                                          key=lambda Layer: Layer[-1],
                                          reverse=True)
            print('list_of_layer RESORTED ! ')
            print(tabulate(All_Gemm_tensor_size, headers=columns))
            print(
                '====================================================================================\n'
            )

    except Exception as e:
        print("Unable to display: " + str(e))

 
#從這裡開始
print("start")
list_tensor_size(inferred_model.graph)