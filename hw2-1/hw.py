from util import *
import torch
import torchvision.models as models
from torchvision.models.googlenet import GoogLeNet_Weights
import torchinfo

# global variables to config...

# define device to cpu/cuda
# will automatically choose to use cuda or cpu
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device)

# load GoogLeNet model
model = models.googlenet(weights=GoogLeNet_Weights.DEFAULT) \
              .to(device) \
              .eval() # locked as evaluation mode

input_shape = (3, 224, 224)

#! 2-1-5 configurable variable
#! forward_input_for_2_1_5 is variable that with forward to model
forward_input_for_2_1_5 = torch.zeros(1, *input_shape)
forward_input_for_2_1_5 = forward_input_for_2_1_5.to(device)
torch.set_printoptions(edgeitems=1) # print less info for output readibility

# install dependency
dep = ["ptflops"]
install_dependency(dep)

# 2-1-1 Calculate model parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print_hw_result("2-1-1",
                "Calculate model parameters",
                f"Total parameters: {total_params}",
                f"trainable parameters: {trainable_params}",
                f"non-trainable parameters: {total_params - trainable_params}")

# 2-1-2 Calculate memory requirements for storing weights
total_param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
print_hw_result("2-1-2",
                "Calculate model memory requirements for storing weights",
                f"Total memory for parameters: {total_param_bytes} bytes")

# 2-1-3. Use Torchinfo to print model architecture
#        including the number of parameters and
#        the output activation size of each layer
model_statistic = torchinfo.summary(model,
                                    input_shape,
                                    batch_dim=0,
                                    col_names=("num_params",
                                                "output_size",),
                                    verbose=0)
print_hw_result("2-1-3",
                "Peek model num_params and output size of each layer use Torchinfo",
                *str(model_statistic).split("\n"))

# 2-1-4. Calculate computation requirement
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
print_hw_result("2-1-4",
                "Calculate computation requirements",
                *info_to_print)

# 2-1-5 Use forward hooks to extract the output activations of the Conv2d layers
info_to_print = []
conv2d_count = 0
focus_layer_name = "Conv2d"

def add_unique_name(module: torch.nn.Module, _, __):
    global conv2d_count
    for m in module.modules():
        if not hasattr(m, 'my_name') and m.__class__.__name__ == focus_layer_name:
            m.my_name = f"{focus_layer_name}-No.{conv2d_count}"
            conv2d_count += 1

def fetch_output_activations(module: torch.nn.Module, _, output: torch.Tensor):
    if focus_layer_name == module.__class__.__name__:
        formatted_str = '\n\t\t'.join(str(output).split('\n'))
        info_to_print.append(f"Module [ {module.my_name:^12} ] has output activation:\n\t\t{formatted_str}\n")

model.apply(lambda m: m.register_forward_hook(add_unique_name) and 
            m.register_forward_hook(fetch_output_activations))
model(forward_input_for_2_1_5)
print_hw_result("2-1-5",
                "Use forward hooks to extract the output activations of the Conv2d layers",
                *info_to_print)