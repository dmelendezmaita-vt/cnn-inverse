"""
Networks with 2D convolutional layers.
"""

import math
import torch
import torch.nn as nn

# --------------------------------------
# Convolutional Nets
# --------------------------------------


class ConvNet(nn.Module):
    r"""
    Network with convolutional layers followed by dense layers.

    Args:
        input_channels: number of input channels
    """

    def __init__(
        self,
        input_channels,
        hidden_conv_layers_channels_mult=[8, 16, 32],
        hidden_conv_layers_kernels=[3, 3, 3],
        hidden_conv_layers_activation=nn.ReLU(),
        hidden_conv_layers_kwargs={},
        hidden_dense_input_size=None,
        hidden_dense_layers_sizes=[],
        hidden_dense_layers_activation=nn.ReLU(),
        hidden_dense_layers_kwargs={},
        output_size=None,
        output_layer_activation=None,
        output_layer_kwargs={},
        use_dropout=False,
    ):
        super().__init__()
        # set from arguments
        self.input_channels = input_channels
        self.hidden_conv_layers_activation = hidden_conv_layers_activation
        self.hidden_dense_layers_activation = hidden_dense_layers_activation
        self.output_layer_activation = output_layer_activation
        if use_dropout:
            self.dropout = nn.Dropout(use_dropout)
        else:
            self.dropout = None
        # create hidden convolutional layers
        assert len(hidden_conv_layers_channels_mult) == len(hidden_conv_layers_kernels)
        in_channels = input_channels
        self.hidden_conv_layers = nn.ModuleList()
        for channel_mult, kernel_size in zip(
            hidden_conv_layers_channels_mult, hidden_conv_layers_kernels
        ):
            out_channels = channel_mult * input_channels
            layer = nn.Conv2d(
                in_channels, out_channels, kernel_size, **hidden_conv_layers_kwargs
            )
            self.hidden_conv_layers.append(layer)
            in_channels = out_channels
        # create hidden dense layers
        assert hidden_dense_input_size is not None or 0 == len(
            hidden_dense_layers_sizes
        )
        in_size = hidden_dense_input_size
        self.hidden_dense_layers = nn.ModuleList()
        for layer_size in hidden_dense_layers_sizes:
            layer = nn.Linear(in_size, layer_size, **hidden_dense_layers_kwargs)
            self.hidden_dense_layers.append(layer)
            in_size = layer_size
        # create output layer
        if in_size is not None and output_size is not None:
            self.output_layer = nn.Linear(in_size, output_size, **output_layer_kwargs)
        else:
            self.output_layer = None
        # initialize parameters
        self.init_parameters()

    def forward(self, x):
        r"""Applies the model function: y = model(x)

        Args:
            x: input tensor
        """
        assert x.size(1) == self.input_channels, f"{x.size(1)=}, {self.input_channels=}"
        h = x
        # apply hidden convolutional layers
        for layer in self.hidden_conv_layers:
            h = layer(h)
            if self.hidden_conv_layers_activation is not None:
                h = self.hidden_conv_layers_activation(h)
            if self.dropout is not None:
                h = self.dropout(h)
        h = torch.flatten(h, 1)
        # apply hidden dense layers
        for layer in self.hidden_dense_layers:
            h = layer(h)
            if self.hidden_dense_layers_activation is not None:
                h = self.hidden_dense_layers_activation(h)
            if self.dropout is not None:
                h = self.dropout(h)
        # apply output layer
        if self.output_layer is not None:
            y = self.output_layer(h)
            if self.output_layer_activation is not None:
                y = self.output_layer_activation(y)
        else:
            y = h
        return y

    def init_parameters(self):
        r"""Initializes the values of trainable parameters."""
        # initialize hidden convolutional layers
        gain = _get_gain(self.hidden_conv_layers_activation)
        for layer in self.hidden_conv_layers:
            _set_init_parameters(layer, gain)
        # initialize hidden dense layers
        gain = _get_gain(self.hidden_dense_layers_activation)
        for layer in self.hidden_dense_layers:
            _set_init_parameters(layer, gain)
        # initialize output layer
        gain = _get_gain(self.output_layer_activation)
        if self.output_layer is not None:
            _set_init_parameters(self.output_layer, gain, bias_scale=0.0)


# --------------------------------------
# Upsample Convolutional Nets
# --------------------------------------


class ConvUpsampleNet_Reshuffle(nn.Module):
    r"""
    Convolutional network for upsampling via reshuffling (or depth-to-space).

    Args:
        input_channels: number of channels
        scale_factor:   factor by which the coarse input gets scaled to produce fine outputs
    """

    def __init__(
        self,
        input_channels,
        scale_factor,
        pre_up_layers_channels=[8, 16, 32],
        pre_up_layers_kernels=[3, 3, 3],
        pre_up_layers_activation=nn.ReLU(),
        pre_up_layers_kwargs={},
        post_up_layers_channels=[],
        post_up_layers_kernels=[],
        post_up_layers_activation=nn.ReLU(),
        post_up_layers_kwargs={},
        output_activation=None,
        use_dropout=False,
    ):
        super().__init__()
        # set from arguments
        self.input_channels = input_channels
        self.scale_factor = scale_factor
        self.pre_up_layers_activation = pre_up_layers_activation
        self.post_up_layers_activation = post_up_layers_activation
        self.output_activation = output_activation
        if use_dropout:
            self.dropout = nn.Dropout(use_dropout)
        else:
            self.dropout = None
        # create hidden convolutional layers before upsampling
        assert len(pre_up_layers_channels) == len(pre_up_layers_kernels)
        in_channels = input_channels
        self.pre_up_layers = nn.ModuleList()
        for channels, kernel_size in zip(pre_up_layers_channels, pre_up_layers_kernels):
            out_channels = channels
            layer = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                **pre_up_layers_kwargs,
                padding=1,
                padding_mode="replicate",
            )
            self.pre_up_layers.append(layer)
            in_channels = out_channels
        # create upsample layer
        self.up_layer = nn.Conv2d(
            in_channels,
            scale_factor**2,
            kernel_size,
            **pre_up_layers_kwargs,
            padding=1,
            padding_mode="replicate",
        )
        # create hidden convolutional layers after upsampling
        assert len(post_up_layers_channels) == len(post_up_layers_kernels)
        in_channels = input_channels
        self.post_up_layers = nn.ModuleList()
        for channels, kernel_size in zip(
            post_up_layers_channels, post_up_layers_kernels
        ):
            out_channels = channels
            layer = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                **post_up_layers_kwargs,
                padding=1,
                padding_mode="replicate",
            )
            self.post_up_layers.append(layer)
            in_channels = out_channels
        # initialize parameters
        self.init_parameters()

    def forward(self, x):
        r"""Applies the network's forward function: y = net(x)

        Args:
            x: input tensor
        """
        assert x.size(1) == self.input_channels
        h = x
        # apply hidden convolutional layers before upsampling
        for layer in self.pre_up_layers:
            h = layer(h)
            if self.pre_up_layers_activation is not None:
                h = self.pre_up_layers_activation(h)
            if self.dropout is not None:
                h = self.dropout(h)
        # apply upsample layer
        h = self.up_layer(h)
        h = nn.functional.pixel_shuffle(h, self.scale_factor)
        # apply hidden convolutional layers after upsampling
        for layer in self.post_up_layers:
            if self.post_up_layers_activation is not None:
                h = self.post_up_layers_activation(h)
            if self.dropout is not None:
                h = self.dropout(h)
            h = layer(h)
        # apply output activation
        if self.output_activation is not None:
            y = self.output_activation(h)
        else:
            y = h
        return y

    def init_parameters(self):
        r"""Initializes the values of trainable parameters."""
        # initialize hidden convolutional layers before upsampling
        gain = _get_gain(self.pre_up_layers_activation)
        for layer in self.pre_up_layers:
            _set_init_parameters(layer, gain)
        # initialize hidden convolutional layers after upsampling
        if self.post_up_layers:
            gain = _get_gain(self.post_up_layers_activation)
            _set_init_parameters(self.up_layer, gain)
            for layer in self.post_up_layers[:-1]:
                _set_init_parameters(layer, gain)
            _set_init_parameters(self.post_up_layers[-1], _get_gain(None))
        else:
            _set_init_parameters(self.up_layer, _get_gain(None))


class ConvUpsampleNet_Interpolate(nn.Module):
    r"""
    Convolutional network for upsampling via interpolation.

    Args:
        input_channels: number of channels
    """

    def __init__(
        self,
        input_channels,
        conv_layers_channels=[4, 4, 1],
        conv_layers_kernels=[3, 3, 3],
        conv_layers_activation=nn.ReLU(),
        conv_layers_kwargs={},
        upsample_layer_indices=[1],
        scale_factor=2,
        interp_mode="nearest",
        output_activation=None,
        use_dropout=False,
    ):
        super().__init__()
        # set from arguments
        self.input_channels = input_channels
        self.upsample_layer_indices = upsample_layer_indices
        self.scale_factor = scale_factor
        self.interp_mode = interp_mode
        self.conv_layers_activation = conv_layers_activation
        self.output_activation = output_activation
        if use_dropout:
            self.dropout = nn.Dropout(use_dropout)
        else:
            self.dropout = None
        # create hidden convolutional layers
        assert len(conv_layers_channels) == len(conv_layers_kernels)
        in_channels = input_channels
        self.conv_layers = nn.ModuleList()
        for channels, kernel_size in zip(conv_layers_channels, conv_layers_kernels):
            out_channels = channels
            layer = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                **conv_layers_kwargs,
                padding=1,
                padding_mode="replicate",
            )
            self.conv_layers.append(layer)
            in_channels = out_channels
        # initialize parameters
        self.init_parameters()

    def forward(self, x):
        r"""Applies the network's forward function: y = net(x)

        Args:
            x: input tensor
        """
        assert x.size(1) == self.input_channels
        h = x
        # apply hidden convolutional layers
        for i, layer in enumerate(self.conv_layers):
            if i in self.upsample_layer_indices:
                h = nn.functional.interpolate(
                    h, scale_factor=self.scale_factor, mode=self.interp_mode
                )
            h = layer(h)
            if (
                self.conv_layers_activation is not None
                and i < len(self.conv_layers) - 1
            ):
                h = self.conv_layers_activation(h)
            if self.dropout is not None:
                h = self.dropout(h)
        # apply output activation
        if self.output_activation is not None:
            y = self.output_activation(h)
        else:
            y = h
        return y

    def init_parameters(self):
        r"""Initializes the values of trainable parameters."""
        gain = _get_gain(self.conv_layers_activation)
        for layer in self.conv_layers[:-1]:
            _set_init_parameters(layer, gain)
        _set_init_parameters(self.conv_layers[-1], _get_gain(None))


# --------------------------------------
# UNet Components
# --------------------------------------


class Downsample(nn.Module):
    """
    A downsampling layer with a convolution.

    Args:
      input_channels   Channels in the inputs.
      output_channels  Channels in the outputs.
    """

    def __init__(
        self,
        input_channels,
        output_channels,
        kernel_size,
        activation=None,
        dropout=None,
        scale_factor=2,
        **conv_kwargs,
    ):
        super().__init__()
        # create convolutional layer with stride=scale_factor
        self.layer = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size,
            **conv_kwargs,
            padding=1,
            padding_mode="replicate",
            stride=scale_factor,
        )
        # set from arguments
        self.input_channels = input_channels
        self.scale_factor = scale_factor
        self.activation = activation
        self.dropout = dropout
        # initialize parameters
        self.init_parameters()

    def forward(self, x):
        assert x.size(1) == self.input_channels
        h = self.layer(x)
        if self.activation is not None:
            h = self.activation(h)
        if self.dropout is not None:
            h = self.dropout(h)
        y = h
        return y

    def init_parameters(self):
        r"""
        Initializes the values of trainable parameters.
        """
        _set_init_parameters(self.layer, _get_gain(self.activation))


class Upsample(nn.Module):
    """
    An upsampling layer with a convolution.

    Args:
      input_channels   Channels in the inputs.
      output_channels  Channels in the outputs.
    """

    def __init__(
        self,
        input_channels,
        output_channels,
        kernel_size,
        activation=None,
        dropout=None,
        scale_factor=2,
        interp_mode="nearest",
        **conv_kwargs,
    ):
        super().__init__()
        # create convolutional layer
        self.layer = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size,
            **conv_kwargs,
            padding=1,
            padding_mode="replicate",
        )
        # set from arguments
        self.input_channels = input_channels
        self.scale_factor = scale_factor
        self.interp_mode = interp_mode
        self.activation = activation
        self.dropout = dropout
        # initialize parameters
        self.init_parameters()

    def forward(self, x):
        assert x.size(1) == self.input_channels
        h = nn.functional.interpolate(
            x, scale_factor=self.scale_factor, mode=self.interp_mode
        )
        h = self.layer(h)
        if self.activation is not None:
            h = self.activation(h)
        if self.dropout is not None:
            h = self.dropout(h)
        y = h
        return y

    def init_parameters(self):
        r"""
        Initializes the values of trainable parameters.
        """
        _set_init_parameters(self.layer, _get_gain(self.activation))


class LevelBlock(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        kernel_size,
        normalization=None,
        activation=None,
        dropout=None,
        **conv_kwargs,
    ):
        super().__init__()
        # set from arguments
        self.input_channels = input_channels
        # create convolutional layer
        self.layer = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size,
            **conv_kwargs,
            padding=1,
            padding_mode="replicate",
        )
        # create residual/skip connection
        if input_channels == output_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(input_channels, output_channels, 1)
        # create additional layers
        self.normalization = normalization
        self.activation = activation
        self.dropout = dropout
        # initialize parameters
        self.init_parameters()

    def forward(self, x):
        assert x.size(1) == self.input_channels
        h = self.layer(x) + self.residual_layer(x)
        if self.normalization is not None:
            h = self.normalization(h)
        if self.activation is not None:
            h = self.activation(h)
        if self.dropout is not None:
            h = self.dropout(h)
        y = h
        return y

    def init_parameters(self):
        r"""Initializes the values of trainable parameters."""
        _set_zero_parameters(self.layer)
        if not isinstance(self.residual_layer, nn.Identity):
            _set_init_parameters(self.residual_layer, _get_gain(None))


def Normalization(num_channels, num_groups=1):
    return nn.GroupNorm(num_groups, num_channels)


# TODO replace by implementation of LevelBlock
class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param output_channels: if specified, the number of out channels.
    :param use_conv: if True and output_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    """

    def __init__(
        self,
        input_channels,
        output_channels=None,
        use_conv=False,
        normalization=None,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels or input_channels
        # create input layers
        self.in_layers = nn.Sequential(
            normalization(input_channels),
            nn.SiLU(),
            nn.Conv2d(
                input_channels,
                self.output_channels,
                3,
                padding=1,
                padding_mode="replicate",
            ),
        )
        # create output layers
        self.out_layers = nn.Sequential(
            normalization(self.output_channels),
            nn.SiLU(),
            _set_zero_parameters(
                nn.Conv2d(
                    self.output_channels,
                    self.output_channels,
                    3,
                    padding=1,
                    padding_mode="replicate",
                )
            ),
        )
        # create skip connection
        if self.output_channels == input_channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv2d(
                input_channels,
                self.output_channels,
                3,
                padding=1,
                padding_mode="replicate",
            )
        else:
            self.skip_connection = nn.Conv2d(input_channels, self.output_channels, 1)

    def forward(self, x):
        """
        Apply the block to a Tensor.

        :param x: an [N x C x ...] Tensor of features.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert x.size(1) == self.input_channels
        h = self.in_layers(x)
        h = self.out_layers(h)
        return self.skip_connection(x) + h


# --------------------------------------
# Utility Functions
# --------------------------------------


def _get_gain(activation):
    r"""Calculates the gain to be used as an argument for initializing parameter values."""
    if activation is not None:
        activation_name = type(activation).__name__.lower()
        if activation_name in ["silu", "gelu"]:
            activation_name = "relu"
        gain = nn.init.calculate_gain(activation_name)
    else:
        gain = nn.init.calculate_gain("conv2d")
    return gain


def _set_init_parameters(layer, gain=1.0, bias_scale=0.1):
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


def _set_zero_parameters(layer):
    r"""
    Zeros the parameters of a layer.
    """
    for p in layer.parameters():
        torch.nn.init.zeros_(p)
    return layer


# --------------------------------------
# Tests
# --------------------------------------

# TODO use doxygen for these test


def test_ConvUpsampleNet_Reshuffle():
    print("---------------------------------------^")
    net = ConvUpsampleNet_Reshuffle(1, 2)
    print(net)

    print("Test 1:")
    x = torch.tensor([[[[1.0, -1.0], [1.0, -1.0]]]])
    y = net(x)
    print("- input  x =", x, sep="\n")
    print("- output y =", y, sep="\n")
    print("---------------------------------------$")


def test_ConvUpsampleNet_Interpolate():
    print("---------------------------------------^")
    net = ConvUpsampleNet_Interpolate(1)
    print(net)

    print("Test 1:")
    x = torch.tensor([[[[1.0, -1.0], [1.0, -1.0]]]])
    y = net(x)
    print("- input  x =", x, sep="\n")
    print("- output y =", y, sep="\n")
    print("---------------------------------------$")


def test_Upsample():
    print("---------------------------------------^")
    layer = Upsample(1, 1, 3)
    print(layer)

    print("Test 1:")
    row = 4 * [1.0]
    x = torch.tensor([[[row for _ in range(4)]]])
    y = layer(x)
    print("- input  x =", x, sep="\n")
    print("- output y =", y, sep="\n")
    print("---------------------------------------$")


def test_Downsample():
    print("---------------------------------------^")
    layer = Downsample(1, 1, 3)
    print(layer)

    print("Test 1:")
    row = 8 * [1.0]
    x = torch.tensor([[[row for _ in range(8)]]])
    y = layer(x)
    print("- input  x =", x, sep="\n")
    print("- output y =", y, sep="\n")
    print("---------------------------------------$")


def test_LevelBlock():
    print("---------------------------------------^")
    layer = LevelBlock(1, 1, 3, activation=nn.ReLU())
    print(layer)

    print("Test 1:")
    row = 8 * [1.0]
    x = torch.tensor([[[row for _ in range(8)]]])
    y = layer(x)
    print("- input  x =", x, sep="\n")
    print("- output y =", y, sep="\n")
    print("---------------------------------------$")


if __name__ == "__main__":
    r"""Runs tests."""
    test_ConvUpsampleNet_Reshuffle()
    test_ConvUpsampleNet_Interpolate()
    test_Upsample()
    test_Downsample()
    test_LevelBlock()
