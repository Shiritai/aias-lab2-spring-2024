from util import *

dep = ["onnx2torch"]
install_dependency(dep)

import onnx
from onnx2torch import convert

# global variables to config...
model_dir = "models"
onnx_model_fn = "model.onnx"
onnx_model_path = f"{model_dir}/{onnx_model_fn}"
torch_script_model_fn = "model.pt"
torch_script_model_path = f"{model_dir}/{torch_script_model_fn}"
torch_model_fn = "model_norm.pt"
torch_model_path = f"{model_dir}/{torch_model_fn}"

# 2-4-0: convert model
if not os.path.exists(onnx_model_path):
    if not os.path.exists(model_dir):
        run_process(["mkdir", "-p", model_dir])
    download_file("https://github.com/onnx/models/raw/main/validated/vision/classification/inception_and_googlenet/googlenet/model/googlenet-12.onnx",
                  onnx_model_path)
else:
    print(f"Model {onnx_model_path} exist, skip downloading")

model = onnx.load(onnx_model_path)

import torch
import torchinfo
from functools import reduce

device = torch.device("cpu")

torch_model = convert(model).to(device)

# Add forward hook to record output model shape
# def activation_recorder(model: torch.nn.Module, _, output: torch.Tensor):
#     model.output_shape = reduce(lambda a, b: a * b, output.shape, 1)
# torch_model.apply(lambda m: m.register_forward_hook(activation_recorder))

torch_model.eval()
input_shape = (1, 3, 224, 224)
torchinfo.summary(torch_model,
                  input_size=input_shape,
                  col_names=["input_size", "output_size", "num_params"],
                  device=device)
torch.save(torch_model, torch_model_path)

example = torch.rand(*input_shape).to(device)
traced_script_module = torch.jit.trace(torch_model, example)
traced_script_module.save(torch_script_model_path)
