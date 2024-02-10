#!/usr/bin/python3

# -----------------------------------------------------------------------
# This script is used to convert a pretrained reset18 pytorch model into 
# a serialized torchscript model file. 
# It will produce a traced_resnet_model.pt file in your working directory
# -----------------------------------------------------------------------

import torch
import torchvision

# An instance of your model.
model = torchvision.models.resnet18()

# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 3, 224, 224)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, example)

# Serializing Your Script Module to a File
traced_script_module.save("traced_resnet_model.pt")

