"""
Model for Autoencoder.
"""

import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, e_net, d_net, output_layer_transformation=None):
        r"""Creates the model.

        Args:
            e_net (Module): encoder network
            d_net (Module): decoder network
        """
        super().__init__()
        self.e_net = e_net
        self.d_net = d_net
        self.output_layer_transformation = output_layer_transformation

    def forward(self, x):
        r"""Applies the model function: y = d_net( e_net(x) )

        Args:
            x (tensor): input tensor
        """
        z = self.e_net(x)
        y = self.d_net(z)
        if self.output_layer_transformation is not None:
            y = self.output_layer_transformation(y)
        return y

    def encode(self, x):
        r"""Applies the encoder: z = e_net(x)

        Args:
            x (tensor): input tensor
        """
        return self.e_net(x)

    def decode(self, z):
        r"""Applies the decoder: y = d_net(z)

        Args:
            z (tensor): input tensor
        """
        y = self.d_net(z)
        if self.output_layer_transformation is not None:
            y = self.output_layer_transformation(y)
        return y
