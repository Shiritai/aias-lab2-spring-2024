from util import *

import onnx
from onnx_helper import OnnxHelper


from dynamic_linear import DynamicLinear, GemmInputInfo, GemmOutputInfo

# global variables to config...
model_dir = "models"
if not os.path.exists(model_dir):
    run_process(["mkdir", "-p", model_dir])

# max width of each dimension
m_mx, n_mx, k_mx = 64, 64, 64

# 2-3-1 + 2-3-2
m, k, n = 32, 64, 128
my_model = DynamicLinear(m,
                         k,
                         n,
                         m_mx,
                         k_mx,
                         n_mx,
                         GemmInputInfo("A"),
                         GemmInputInfo("B"),
                         GemmInputInfo("bias"),
                         GemmOutputInfo("C")
                         )

# 2-3-1
test_model_path = f"{model_dir}/subgraph1.onnx"
onnx.save(my_model.test_model, test_model_path)
test_helper = OnnxHelper(my_model.test_model, to_check=True)
print_hw_result("2-3-1",
                "Create a subgraph (1) that consist of a single Linear layer of size MxKxN",
                *json_stringify(test_helper.summary(wrap_graph_name=True, override_name="Subgraph 1")))

# 2-3-2
dl_model_path = f"{model_dir}/subgraph2.onnx"
onnx.save(my_model.model, dl_model_path)
dl_helper = OnnxHelper(my_model.model)
print_hw_result("2-3-2",
                "Create a subgraph (2) as shown in the above diagram for the subgraph (1)",
                *json_stringify(dl_helper.summary(wrap_graph_name=True, override_name="Subgraph 2")))

# 2-3-3
# fetch alexnet model
from torchvision import models
import torch

alexnet_model_path = f"{model_dir}/model.onnx"
# alexnet_model_url = "https://github.com/onnx/models/raw/main/validated/vision/classification/alexnet/model/bvlcalexnet-12.onnx"
if not os.path.exists(alexnet_model_path):
    if not os.path.exists(model_dir):
        run_process(["mkdir", "-p", model_dir])
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Download alexnet from torchvision, convert to ONNX model and save it
    #   ref: https://pytorch.org/docs/stable/onnx_torchscript.html
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    alexnet_model = models.alexnet().to(device)
    alexnet_model.eval() # Model must be in evaluation mode for export

    # Providing input and output names sets the display names for values
    # within the model's graph. Setting these does not change the semantics
    # of the graph; it is only for readability.
    #
    # The inputs to the network consist of the flat list of inputs (i.e.
    # the values you would pass to the forward() method) followed by the
    # flat list of parameters. You can partially specify names, i.e. provide
    # a list here shorter than the number of inputs to the model, and we will
    # only set that subset of names, starting from the beginning.
    input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
    output_names = [ "output1" ]

    torch.onnx.export(alexnet_model,
                    dummy_input,
                    alexnet_model_path,
                    verbose=True,
                    input_names=input_names,
                    output_names=output_names)
else:
    print(f"Model {alexnet_model_path} exist, skip downloading")

# Alexnet: load and initialize helper
alexnet_model = onnx.load(alexnet_model_path)
alexnet_helper = OnnxHelper(alexnet_model, to_check=True)
print_hw_result("2-3-0",
                "Peek converted Alexnet",
                *json_stringify(alexnet_helper.summary(wrap_graph_name=True, override_name="Alexnet")))

# Alexnet (to be modified): load and initialize helper
my_alexnet_model = onnx.load(alexnet_model_path)
my_alexnet_helper = OnnxHelper(my_alexnet_model, to_check=True)

def parse_gemm_size(info: dict[OnnxHelper.NodeInfoType, dict]) \
    -> tuple[tuple[tuple[str, ],
                   tuple[str, ],
                   tuple[str, ],
                   tuple[str, ]],
             tuple[bool, bool],
             tuple[int, int, int]]:
    """
    Parse and return m, k, n in Gemm operation
    """
    # print("\n".join(json_stringify(info)))
    
    input_info = info["input"]
    output_info = info["output"]
    C = extract_dict(output_info)
    output_name, output_data = C
    output_shape: tuple = output_data["shape"]

    # find A, B, bias
    A, B, bias = (), (), ()
    for entry in input_info.items():
        # print("entry shape", len(entry[1]["shape"]))
        if "output" in entry[0]:
            A = entry
        elif len(entry[1]["shape"]) == 1:
            bias = entry
        else:
            B = entry
    # find m, n, k
    m, n = output_shape
    k = A[1]["shape"][0 if A[1]["shape"][0] not in output_shape else 1]

    return ((A, B, bias, C),
            (info["attribute"].get("transA", False),
             info["attribute"].get("transB", False)),
            (m, k, n))
    

# Manipulate onnx model using OnnxHelper API implement by myself
replacements: List[tuple[int, onnx.GraphProto]] = []
my_models: List[DynamicLinear] = []
# Prepare replacement layers
for idx in alexnet_helper.select_all(lambda n: n.op_type == "Gemm"):
    _, node_info = alexnet_helper.get_info_from_node(idx)
    (A, B, bias, C), (trans_a, trans_b), (m, k, n) = parse_gemm_size(node_info)

    a_name, b_name, bias_name, c_name = A[0], B[0], bias[0], C[0]

    model = DynamicLinear(m,
                          k,
                          n,
                          m_mx,
                          k_mx,
                          n_mx,
                          a_info=GemmInputInfo(a_name, trans_a),
                          b_info=GemmInputInfo(b_name, trans_b),
                          bias_info=GemmInputInfo(bias_name),
                          c_info=GemmOutputInfo(c_name),
                          name=str(idx),
                          named_ls=my_alexnet_helper.get_named_list()
                          )
    
    # this is for sure that my model works gracefully
    model_helper = OnnxHelper(model.model)
    
    replacements.append((idx, model.graph))
    my_models.append(model)

# Replace all GEMM layers at correct place
my_alexnet_helper.replace_all(replacements)
onnx.save(my_alexnet_helper.model, f"{model_dir}/modified.onnx") # for viewing result

print_hw_result("2-3-3",
                "Replace the Linear layers in the AlexNet with the equivalent subgraphs (2)",
                *json_stringify(my_alexnet_helper.summary(wrap_graph_name=True)))
    
# 2-3-4 Correctness Verification

run_cases = 0
for m in my_models:
    # check m is an valid graph
    m_helper = OnnxHelper(m.test_model)
    m_helper = OnnxHelper(m.model)
    # run test
    run_cases = m.test_equivalence()

print_hw_result("2-3-4",
                "Correctness Verification",
                f"Passed {run_cases} random test cases for each part of node replacement")