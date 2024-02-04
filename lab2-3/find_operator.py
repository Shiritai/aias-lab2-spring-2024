import onnx
from onnx import shape_inference
import sys
from tabulate import tabulate
from onnx import onnx_ml_pb2 as xpb2

onnx_model = onnx.load("lenet.onnx", load_external_data=False)
onnx.checker.check_model(onnx_model)

inferred_model = shape_inference.infer_shapes(onnx_model)
print('shape inference complete ...')


def _parse_element(elem: xpb2.ValueInfoProto):
    name = getattr(elem, 'name', "None")
    shape_str = "NA"
    etype = getattr(elem, 'type', False)
    if etype:
        ttype = getattr(etype, 'tensor_type', False)
        if ttype:
            shape = getattr(elem.type.tensor_type, "shape", False)
            if shape:
                shape_str = "["
                dims = getattr(shape, 'dim', [])
                for dim in dims:
                    vals = getattr(dim, 'dim_value', "?")
                    shape_str += (str(vals) + ",")
                shape_str = shape_str.rstrip(",")
                shape_str += "]"
    return name, shape_str
def find_operator(graph: xpb2.GraphProto):
    try:
        for i, node in enumerate(inferred_model.graph.node):
                if node.name == "":
                    inferred_model.graph.node[i].name = str(i)
        if type(graph) is not xpb2.GraphProto:
            sys.exit('The input graph is not a GraphProto!')

        node_nList = [k.name for k in graph.node]
        op_dict = {}
        for node in graph.node:
            if node.op_type in op_dict:
                op_dict[node.op_type][node.name] = node_nList.index(node.name)
            else:
                op_dict[node.op_type] = {
                    node.name: node_nList.index(node.name)
                } 

        # init the list
        list_of_layer = []
        list_of_unknown_tensor = []

        # get the occur #, total data element
        for op_type in op_dict:
            occur_num = len(op_dict[op_type])
            total_data_elem = int(0)
            unknown_tensor_list = []

            for op in op_dict[op_type]:
                op_data_elem = int(0)
                for input_name in graph.node[op_dict[op_type][op]].input:
                    input_nlist = [k.name for k in graph.input]
                    initializer_nlist = [k.name for k in graph.initializer]
                    value_info_nlist = [k.name for k in graph.value_info]
                    output_nlist = [k.name for k in graph.output]

                    # get tensor data
                    if input_name in input_nlist:
                        idx = input_nlist.index(input_name)
                        proto= graph.input[idx]
                        data_type = int(1)
                    elif input_name in value_info_nlist:
                        idx = value_info_nlist.index(input_name)
                        proto= graph.value_info[idx]
                        data_type = int(2)
                    elif input_name in initializer_nlist:
                        idx = initializer_nlist.index(input_name)
                        proto= graph.initializer[idx]
                        data_type = int(3)
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
                    if proto:
                        input_tensor_size = int(1)
                        if data_type == 1 or data_type == 2:
                            name, shape_str = _parse_element(proto)
                            shape_str = shape_str.strip('[]')
                            shape_str = shape_str.split(',')
                            shape = []
                            for dim in shape_str:
                                input_tensor_size *=int(dim)
                        elif data_type == 3:
                            # get the shape of the tensor
                            shape = getattr(proto, 'dims', [])
                            for dim in shape:
                                input_tensor_size *= dim
                        else:
                            print(
                                '[unexpected error] in [get_info] add_up_total_data_elem'
                            )
                    else:
                        print("Can't find the input ", input_name, " of the operator ",
                            op, 'SKIP IT !')
                        unknown_tensor_list.append(
                            (op, input_name, graph.node[op_dict[op_type][op]].op_type))
                        continue

                    op_data_elem += input_tensor_size

                total_data_elem += op_data_elem
            list_of_layer.append((op_type, occur_num, total_data_elem))
            list_of_unknown_tensor.extend(unknown_tensor_list)
        print('list_of_layer BUILT ! ')

        # resort the list
        list_of_layer = sorted(list_of_layer,
                               key=lambda Layer: Layer[2],
                               reverse=True)
        print('list_of_layer RESORTED ! ')

        # Display result
        columns = ['op_type', 'occur #', 'Total data elements #']
        print(tabulate(list_of_layer, headers=columns))
        print(
            '====================================================================================\n'
        )

        columns = ['op_name', 'unfound_tensor', 'op_type']
        print(tabulate(list_of_unknown_tensor, headers=columns))
        print(
            '====================================================================================\n'
        )

        print('DISPLAY successfully ! ')

    except Exception as e:
        print("Unable to display: " + str(e))
        return False

    return True

#從這裡開始
print("start")
find_operator(inferred_model.graph)



