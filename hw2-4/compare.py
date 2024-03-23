from util import *
import torch
import torchinfo

# global variables to config...
model_dir = "models"
torch_model_fn = "model_norm.pt"
torch_model_path = f"{model_dir}/{torch_model_fn}"

# define device to cpu/cuda
# will automatically choose to use cuda or cpu
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
device = torch.device(device)

# load GoogLeNet model
model = torch.load(torch_model_path)

input_shape = (3, 224, 224)

#! 2-1-5 configurable variable
#! forward_input_for_2_1_5 is variable that with forward to model
forward_input_for_2_1_5 = torch.zeros(1, *input_shape)
forward_input_for_2_1_5 = forward_input_for_2_1_5.to(device)
torch.set_printoptions(edgeitems=1) # print less info for output readibility

# install dependency
dep = ["ptflops"]
install_dependency(dep)

total_param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
print_hw_result("2-4-1-compare",
                "Calculate model memory requirements for storing weights",
                f"Total memory for parameters: {total_param_bytes} bytes")

model_statistic = torchinfo.summary(model,
                                    input_shape,
                                    batch_dim=0,
                                    col_names=("num_params",
                                               "output_size",),
                                    verbose=0)
print_hw_result("2-4-0-compare",
                "Peek model num_params and output size of each layer use Torchinfo",
                *str(model_statistic).split("\n"))

info_to_print = []

# Use open source project
#   https://github.com/sovrasov/flops-counter.pytorch.git
from ptflops import get_model_complexity_info

total_macs, _ = get_model_complexity_info(model,
                                          input_shape,
                                          output_precision=10,
                                          print_per_layer_stat=False,
                                          ignore_modules=[torch.nn.Dropout,],)

info_to_print.append(f"Total MACs is {total_macs}")

def get_children(m) -> List[Union[List, str]]:
    """
    Get listed layers of a pytorch model
    """
    if next(m.children(), None) == None:
        return m.__class__
    else:
        return [get_children(c) for c in m.children()]

layers = { *flatten(get_children(model)) }
default_layer_macs, _ = get_model_complexity_info(model,
                                               input_shape,
                                               output_precision=10,
                                               ignore_modules=[*layers],
                                               as_strings=False,
                                               print_per_layer_stat=False,)
print(f"null layer macs: ", default_layer_macs)
print(f"layers: ", layers)
layers = layers - { torch.nn.Dropout }
# iterate sorted set to ensure that anytime the result order is the same
for layer in sorted(layers, key=lambda e: e.__name__):
    layer_macs, _ = get_model_complexity_info(model,
                                              input_shape,
                                              output_precision=10,
                                              ignore_modules=[*(layers-{layer})],
                                              as_strings=False,
                                              print_per_layer_stat=False,)
    layer_macs -= default_layer_macs
    info_to_print.append(f"MACs for [ {layer.__name__:^17} ] layers is {layer_macs:>11}")
print_hw_result("2-4-3-compare",
                "Calculate computation requirements",
                *info_to_print)