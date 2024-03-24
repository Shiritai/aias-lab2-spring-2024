from tabulate import tabulate
from util import *

model_path_prefix = "models"
model_path = f"{model_path_prefix}/model.onnx"
if not os.path.exists(model_path):
    if not os.path.exists(model_path_prefix):
        run_process(["mkdir", "-p", model_path_prefix])
    download_file("https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-10.onnx",
                  model_path)
else:
    print(f"Model {model_path} exist, skip downloading")

# 2-2-1 Model characteristics
import onnx
from onnx_helper import OnnxHelper

model = onnx.load(model_path)
helper = OnnxHelper(model)

unique_ops = { node.op_type for node in model.graph.node }
model_op_list = []

info_list = [
    "Unique operators:",
    # use sorted to make sure result is consistent anytime
    "\t" + "\n\t\t".join(sorted(unique_ops)),
    "",
    "Attribute range of all the operators:",
    *json_stringify({ "GoogleNet": helper.summary() }, indent=1),
]
print_hw_result("2-2-1",
                "Model characteristics",
                *info_list)

# 2-2-{2,3} Bandwidth requirement
bw_ls = [] # memory bandwidth list
max_bw = 0
tot_bw, tot_i_bw, tot_o_bw = 0, 0, 0
i_bw, o_bw = 0, 0 # input and output bandwidth

for cnt, n in enumerate(model.graph.node):
    i_bw, o_bw = 0, 0 # initialize io bw
    if len(n.input) > 0:
        sz = helper.fetch_size(helper.fetch_proto(n.input[0]))
        i_bw += sz["memory_size"]
    if len(n.output) > 0:
        sz = helper.fetch_size(helper.fetch_proto(n.output[0]))
        o_bw += sz["memory_size"]
    if cnt == 0: # only reserve first input size to total bandwidth
        tot_bw += i_bw
    max_bw = max(max_bw, i_bw + o_bw)
    tot_bw += o_bw
    tot_i_bw += i_bw
    tot_o_bw += o_bw
    bw_ls.append([n.name, i_bw, o_bw, i_bw + o_bw])
bw_ls.append([*["-----"] * 4]) # splitter of table
bw_ls.append(["Total I/O activation", tot_i_bw, tot_o_bw, "-"])

info_list = [
    f"Peek bandwidth (2-2-2): {max_bw} bytes",
    f"Total activation bandwidth (2-2-3): {tot_o_bw} bytes",
    f"Total bandwidth: {tot_bw} bytes\n",
    "List of bandwidth of each layer:\n",
    *tabulate(bw_ls,
              tablefmt="rounded_outline",
              colalign=["left", *(["right"] * 3)],
              headers=["layer",
                       "input bw",
                       "output bw",
                       "bandwidth"]).split("\n"),
]

print_hw_result("2-2-{2_3}",
                "Bandwidth requirement",
                *info_list)
