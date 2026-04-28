"""
Utility Functions
"""

import math
from prettytable import PrettyTable
import torch
import torch.nn as nn


def get_gain(activation):
    r"""Calculates the gain to be used as an argument for initializing parameter values."""
    if activation is not None:
        activation_name = type(activation).__name__.lower()
        if activation_name in ["silu", "gelu"]:
            activation_name = "relu"
        gain = nn.init.calculate_gain(activation_name)
    else:
        gain = nn.init.calculate_gain("linear")
    return gain


def set_init_parameters(layer, gain=1.0, bias_scale=0.1):
    r"""
    Initializes the trainable parameters of a layer.

    Args:
        layer:      Layer of a network to be initialized (of type nn.Module)
        gain:       Gain to use for sampling initial parameters
        bias_scale: Scaling of uniform distribution for initializing the bias
    """
    nn.init.xavier_uniform_(layer.weight, gain=gain)
    if layer.bias is not None:
        lim = bias_scale * gain / math.sqrt(layer.bias.size(0))
        nn.init.uniform_(layer.bias, a=-lim, b=+lim)


def set_zero_parameters(layer):
    r"""
    Zeros the parameters of a layer.
    """
    for p in layer.parameters():
        torch.nn.init.zeros_(p)
    return layer


def get_parameters(net):
    r"""
    Original source: https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model
    """
    table = PrettyTable(["Module name", "Num. parameters", "Trainable"])
    table.align["Module name"] = "l"
    table.align["Num. parameters"] = "r"
    table.align["Trainable"] = "c"
    n_trainable_params = 0
    n_nontrainable_params = 0
    for name, parameter in net.named_parameters():
        n_params = parameter.numel()
        if parameter.requires_grad:
            table.add_row([name, n_params, True])
            n_trainable_params += n_params
        else:
            table.add_row([name, n_params, False])
            n_nontrainable_params += n_params
    table.add_divider()
    table.add_row(
        ["Total number of trainable parameters", f"{n_trainable_params}", True]
    )
    table.add_row(
        ["Total number of non-trainable parameters", f"{n_nontrainable_params}", False]
    )
    return n_trainable_params, n_nontrainable_params, table


def count_trainable_parameters(net):
    r"""
    Counts the number of trainable parameters of a network.
    """
    assert isinstance(net, nn.Module)
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def count_all_parameters(net):
    r"""
    Counts the number of all parameters (including non-trainable) of a network.
    """
    assert isinstance(net, nn.Module)
    return sum(p.numel() for p in net.parameters())
