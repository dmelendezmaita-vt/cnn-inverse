"""
Networks with 1D convolutional layers.
"""

from collections import OrderedDict
import logging, math
import torch
import torch.nn as nn

from dlkit.nets.mlp import MLPResNet

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
        hidden_pool_layers_type=None,
        hidden_pool_layers_kernels=None,
        hidden_pool_layers_kwargs={},
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
            layer = nn.Conv1d(
                in_channels, out_channels, kernel_size, **hidden_conv_layers_kwargs
            )
            self.hidden_conv_layers.append(layer)
            in_channels = out_channels
        # create optional pooling layers (e.g., AvgPool1d for paper-match setup)
        self.hidden_pool_layers = None
        if hidden_pool_layers_type is not None:
            if hidden_pool_layers_kernels is None:
                hidden_pool_layers_kernels = len(self.hidden_conv_layers) * [2]
            assert len(hidden_pool_layers_kernels) == len(self.hidden_conv_layers)
            pool_type = str(hidden_pool_layers_type).casefold()
            if pool_type in ["avg", "avgpool", "avgpool1d"]:
                pool_fn = nn.AvgPool1d
            elif pool_type in ["max", "maxpool", "maxpool1d"]:
                pool_fn = nn.MaxPool1d
            else:
                raise ValueError(
                    f"Unknown pool type for ConvNet: {hidden_pool_layers_type}"
                )
            self.hidden_pool_layers = nn.ModuleList()
            for kernel_size in hidden_pool_layers_kernels:
                if kernel_size is None:
                    self.hidden_pool_layers.append(nn.Identity())
                else:
                    self.hidden_pool_layers.append(
                        pool_fn(kernel_size, **hidden_pool_layers_kwargs)
                    )
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
        for i, layer in enumerate(self.hidden_conv_layers):
            h = layer(h)
            if self.hidden_conv_layers_activation is not None:
                h = self.hidden_conv_layers_activation(h)
            if self.dropout is not None:
                h = self.dropout(h)
            if self.hidden_pool_layers is not None:
                h = self.hidden_pool_layers[i](h)
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


class ConvResNet(nn.Module):
    r"""
    Network with residual convolutional layers followed by residual dense layers.

    Args:
        input_channels: number of input channels
        conv_resnet_params: configuration for convolutional layers
        mlp_resnet_params: parameters to pass to MLPResNet constructor

    Specs:
    - doc/specify/2025-10-27a.md
    """

    def __init__(
        self,
        input_channels,
        conv_resnet_params={},
        mlp_resnet_params={},
        with_Conv=nn.Conv1d,
    ):
        super().__init__()
        # set from arguments
        self.input_channels = input_channels
        self.conv_resnet_params = dict(conv_resnet_params)
        self.mlp_resnet_params = dict(mlp_resnet_params)

        # set default convolution parameters
        self.conv_resnet_params.setdefault("channels_mult", [8, 16, 32])
        self.conv_resnet_params.setdefault("kernels", [5, 5, 5])
        self.conv_resnet_params.setdefault("activation", nn.ReLU())
        self.conv_resnet_params.setdefault("use_dropout", False)
        self.conv_resnet_params.setdefault("mlb_kwargs", {})
        assert len(self.conv_resnet_params["channels_mult"]) == len(
            self.conv_resnet_params["kernels"]
        )

        # set scale factor
        if "stride" not in self.conv_resnet_params["mlb_kwargs"]:
            scale_factor = 0.5  # downsample by factor 1/2
        else:
            scale_factor = None

        # set activation
        activation = self.conv_resnet_params["activation"]

        # set up dropout
        if self.conv_resnet_params["use_dropout"]:
            dropout = nn.Dropout(use_dropout)
        else:
            dropout = None

        # create input layer
        in_channels = self.input_channels
        out_channels = self.conv_resnet_params["channels_mult"][0] * self.input_channels
        kernel_size = self.conv_resnet_params["kernels"][0]
        self.input_layer = with_Conv(in_channels, out_channels, 1, groups=in_channels)
        in_channels = out_channels

        # create convolutional residual blocks using MultiLevelBlock
        layers = list()
        for mult, kernel_size in zip(
            self.conv_resnet_params["channels_mult"], self.conv_resnet_params["kernels"]
        ):
            # create normalization
            normalization = nn.GroupNorm(self.input_channels, in_channels)
            # create convolution block
            out_channels = mult * self.input_channels
            layers.append(
                MultiLevelBlock(
                    in_channels,
                    kernel_size,
                    normalization=normalization,
                    normalization_layer_channels=in_channels,
                    activation=activation,
                    activation_layer_channels=4 * in_channels,
                    output_channels=out_channels,
                    dropout=dropout,
                    scale_factor=scale_factor,
                    skip_connection=True,
                    **self.conv_resnet_params["mlb_kwargs"],
                )
            )
            in_channels = out_channels
        self.conv_resnet = nn.Sequential(*layers)

        # create dense layers using MLPResNet if parameters provided
        if self.mlp_resnet_params:
            # check if input_size is provided in self.mlp_resnet_params
            if "input_size" not in self.mlp_resnet_params:
                raise ValueError(
                    "mlp_resnet_params must have 'input_size' (flattened conv output size)"
                )
            self.mlp_resnet = MLPResNet(**self.mlp_resnet_params)
            ###DEV
            # from dlkit.nets.mlp import LinearFiber
            # # create contraction layer for space
            # if isinstance(self.mlp_resnet_params["input_size"], int):
            #     in_fiber_size = self.mlp_resnet_params["input_size"] // out_channels
            # else:
            #     in_fiber_size = self.mlp_resnet_params["input_size"][0] // out_channels
            # self.mlp_space_l0 = LinearFiber(
            #     ndim=2,
            #     input_fiber_size=in_fiber_size,
            #     output_fiber_size=128,
            # )
            # self.mlp_activation = activation
            # self.mlp_space_l1 = LinearFiber(
            #     ndim=2,
            #     input_fiber_size=128,
            #     output_fiber_size=1,
            # )
            # if isinstance(self.mlp_resnet_params["input_size"], int):
            #     in_size = out_channels
            # else:
            #     in_size = (
            #         out_channels +
            #         self.mlp_resnet_params["residual_blocks_sizes"][0][0] -
            #         self.mlp_resnet_params["input_size"][1]
            #     )
            # out_size = self.mlp_resnet_params["output_size"]
            # self.mlp_output_l0 = nn.Linear(in_size, 128)
            # self.mlp_output_l1 = nn.Linear(128, out_size)
            ###/DEV
        else:
            self.mlp_resnet = None

        # initialize parameters
        self.init_parameters()

    def forward(self, x, **h_kwargs):
        r"""Applies the forward function: y = net(x, h0=hidden_input_0, h1=hidden_input1, ...)

        Args:
            x (tensor): input tensor
            h0, h1, ... (tensor, optional): input tensors to hidden dense layers
        """
        assert x.size(1) == self.input_channels, f"{x.size(1)=}, {self.input_channels=}"

        # apply input layer
        h = self.input_layer(x)

        # apply convolutional layers
        for layer in self.conv_resnet:
            h = layer(h)

        # return if nothing to do
        if self.mlp_resnet is None:
            return h

        # flatten for dense layers
        h = torch.flatten(h, 1)

        # apply dense residual network if configured
        y = self.mlp_resnet(h, **h_kwargs)

        ###DEV
        # h = self.mlp_space_l1(self.mlp_activation(self.mlp_space_l0(h)))
        # h = torch.flatten(h, 1)
        # h_in = h_kwargs.get(f"h{0}")
        # if h_in is None:
        #     h_in = h_kwargs.get("h_all")
        # if h_in is not None:
        #     h_in = torch.flatten(h_in, 1)
        #     h = torch.cat([h, h_in], dim=1)
        # y = self.mlp_output_l1(self.mlp_activation(self.mlp_output_l0(h)))
        ###/DEV

        return y

    def init_parameters(self):
        r"""Initializes the values of trainable parameters."""
        # initialize input layer
        _set_init_parameters(self.input_layer, _get_gain(None))
        # initialize convolutional block
        for layer in self.conv_resnet:
            layer.init_parameters()
        # initialize dense block
        if self.mlp_resnet is not None:
            self.mlp_resnet.init_parameters()


# --------------------------------------
# UNet Components
# --------------------------------------


class UNetDownsample(nn.Module):
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
        **layer_kwargs,
    ):
        super().__init__()
        # create convolutional layer with stride=scale_factor
        # add default values only if keys don't exist
        self.layer_kwargs = dict(layer_kwargs)  # copy to avoid modifying input
        self.layer_kwargs.setdefault("padding", 1)
        self.layer_kwargs.setdefault("padding_mode", "replicate")
        self.layer_kwargs.setdefault("stride", scale_factor)
        self.input_channels = input_channels
        # create layers
        block = OrderedDict()
        block["layer"] = nn.Conv1d(
            input_channels,
            output_channels,
            kernel_size,
            **self.layer_kwargs,
        )
        if activation is not None:
            block["activation"] = activation
        if dropout is not None:
            block["dropout"] = dropout
        self.block = nn.Sequential(block)
        # initialize parameters
        self.init_parameters()

    def forward(self, x):
        assert x.size(1) == self.input_channels, f"{x.size(1)=}, {self.input_channels=}"
        y = self.block(x)
        return y

    def init_parameters(self):
        r"""
        Initializes the values of trainable parameters.
        """
        _set_init_parameters(self.block.layer, _get_gain(self.block.activation))


class UNetUpsample(nn.Module):
    """
    An upsampling layer with a convolution.

    Args:
      input_channels   Channels in the inputs.
      output_channels  Channels in the outputs.

    Note:
      Mode `mode='nearest-exact'` matches Scikit-Image and PIL nearest neighbours
      interpolation algorithms and fixes known issues with `mode='nearest'`.
      (https://docs.pytorch.org/docs/2.9/generated/torch.nn.functional.interpolate.html)
    """

    def __init__(
        self,
        input_channels,
        output_channels,
        kernel_size,
        activation=None,
        dropout=None,
        scale_factor=2,
        interp_mode="nearest-exact",
        **layer_kwargs,
    ):
        super().__init__()
        # create convolutional layer
        # add default values only if keys don't exist
        self.layer_kwargs = dict(layer_kwargs)  # copy to avoid modifying input
        self.layer_kwargs.setdefault("padding", 1)
        self.layer_kwargs.setdefault("padding_mode", "replicate")
        self.input_channels = input_channels
        self.scale_factor = scale_factor
        self.interp_mode = interp_mode
        # create layers
        block = OrderedDict()
        block["layer"] = nn.Conv1d(
            input_channels,
            output_channels,
            kernel_size,
            **self.layer_kwargs,
        )
        if activation is not None:
            block["activation"] = activation
        if dropout is not None:
            block["dropout"] = dropout
        self.block = nn.Sequential(block)
        # initialize parameters
        self.init_parameters()

    def forward(self, x):
        assert x.size(1) == self.input_channels, f"{x.size(1)=}, {self.input_channels=}"
        h = nn.functional.interpolate(
            x, scale_factor=self.scale_factor, mode=self.interp_mode
        )
        y = self.block(h)
        return y

    def init_parameters(self):
        r"""
        Initializes the values of trainable parameters.
        """
        _set_init_parameters(self.block.layer, _get_gain(self.block.activation))


def Normalization(num_channels, num_groups=1):
    return nn.GroupNorm(num_groups, num_channels)


class UNetResBlock(nn.Module):
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
            nn.Conv1d(
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
                nn.Conv1d(
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
            self.skip_connection = nn.Conv1d(
                input_channels,
                self.output_channels,
                3,
                padding=1,
                padding_mode="replicate",
            )
        else:
            self.skip_connection = nn.Conv1d(input_channels, self.output_channels, 1)

    def forward(self, x):
        """
        Apply the block to a Tensor.

        :param x: an [N x C x ...] Tensor of features.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert x.size(1) == self.input_channels, f"{x.size(1)=}, {self.input_channels=}"
        h = self.in_layers(x)
        h = self.out_layers(h)
        return self.skip_connection(x) + h


# --------------------------------------
# Universal-Design Multi-Level Components
# --------------------------------------


class MultiLevelBlock(nn.Module):
    """
    A multi-level block with convolutional layers following a universal design.

    Args:
      input_channels   Channels in the inputs.
      output_channels  Channels in the outputs.
    """

    def __init__(
        self,
        input_channels,
        kernel_size,
        normalization=None,
        normalization_layer_channels=None,
        activation=None,
        activation_layer_channels=None,
        output_channels=None,
        dropout=None,
        scale_factor=None,
        skip_connection=False,
        interp_mode="nearest-exact",
        logger=logging.getLogger("dlkit.nets.conv1d.MultiLevelBlock"),
        with_Conv=nn.Conv1d,
        **conv_kwargs,
    ):
        super().__init__()
        self.conv_kwargs = dict(conv_kwargs)  # copy to avoid modifying input args
        # set default padding mode (if key does not exist)
        self.conv_kwargs.setdefault("padding_mode", "replicate")
        # set default padding size (if key does not exist)
        # Note: Conv1d expects an int padding; a 2-tuple causes channel padding
        #   when padding_mode != 'zeros'; use symmetric integer padding for the conv
        #   layer and keep the 2-tuple separately for size bookkeeping
        self.conv_kwargs.setdefault("padding", kernel_size // 2)
        assert (
            isinstance(self.conv_kwargs["padding"], int)
        ), f"Expected type int, got {type(self.conv_kwargs['padding'])}"
        # store a 2-tuple version of padding for internal calculations (left, right)
        self._padding = (self.conv_kwargs["padding"], self.conv_kwargs["padding"])
        # define the padding s.t. `input_size == output_size` after convolution
        self._padding_const_size = (
            (kernel_size - 1) // 2,
            (kernel_size - 1) // 2 + (1 - kernel_size % 2),
        )
        # set attributes from arguments
        self.input_channels = input_channels
        self.interp_mode = interp_mode
        if scale_factor is not None:
            assert (
                "stride" not in conv_kwargs or scale_factor == conv_kwargs["stride"]
            ), f"Invalid args: Cannot set both scale_factor={scale_factor} and stride={conv_kwargs['stride']}."
            if 1 <= scale_factor:
                self.conv_kwargs["stride"] = 1
            elif 0 < scale_factor:
                self.conv_kwargs["stride"] = int(1 / scale_factor)
            else:
                raise ValueError(
                    f"Invalid arg: scale_factor={scale_factor} cannot be non-positive."
                )
            self.scale_factor = float(scale_factor)
        elif "stride" in conv_kwargs:
            self.scale_factor = 1.0 / conv_kwargs["stride"]
        else:
            self.scale_factor = 1.0
        # init channels
        ch = [input_channels]
        # set up normalization
        if normalization_layer_channels or normalization_layer_channels is None:
            if normalization_layer_channels is None:
                normalization_layer_channels = ch[-1]
            ch.append(normalization_layer_channels)
        if normalization is None:
            normalization = nn.GroupNorm(1, ch[-1])
        if (not normalization) != (not normalization_layer_channels):
            logger.warning(
                f"Incoherent args: {normalization=}, {normalization_layer_channels=}"
            )
        # set up activation
        if activation_layer_channels or activation_layer_channels is None:
            if activation_layer_channels is None:
                activation_layer_channels = 4 * ch[-1]
            ch.append(activation_layer_channels)
        if activation is None:
            activation = nn.ReLU()
        # set output channels
        if output_channels is not None:
            ch.append(output_channels)
        else:
            ch.append(input_channels)
        # create layers
        i = 0
        block = OrderedDict()
        block["layer_0"] = with_Conv(ch[i], ch[i + 1], kernel_size, **self.conv_kwargs)
        i += 1
        if normalization:
            block["normalization"] = normalization
        if normalization_layer_channels:
            block["layer_1"] = with_Conv(ch[i], ch[i + 1], 1)
            i += 1
        if activation:
            block["activation"] = activation
        if dropout:
            block["dropout"] = dropout
        if activation_layer_channels:
            block["layer_2"] = with_Conv(ch[i], ch[i + 1], 1)
            i += 1
        self.block = nn.Sequential(block)
        # create skip connection
        if skip_connection:
            self.skip_interp_mode = interp_mode
            if ch[0] == ch[-1]:
                self.skip_connection = nn.Identity()
            else:
                self.skip_connection = with_Conv(ch[0], ch[-1], 1)
        else:
            self.skip_connection = None
        # initialize parameters
        self.init_parameters()

    def forward(self, *x):
        channel_dim, size_dim = 1, 2
        # concatenate along channel dimension
        if 1 < len(x):
            h = torch.cat(x, dim=channel_dim)
        else:
            assert 1 == len(x)
            h = x[0]
        assert (
            h.size(channel_dim) == self.input_channels
        ), f"Expect {self.input_channels=}, got {h.size(channel_dim)=}"
        # scale up
        if 1.0 < self.scale_factor:
            h = nn.functional.interpolate(
                h, scale_factor=self.scale_factor, mode=self.interp_mode
            )
        # apply block
        hb = self.block(h)
        # apply skip connection (optional)
        if self.skip_connection is not None:
            # scale down
            if self.scale_factor < 1.0:
                hs_size = (
                    int(math.ceil(h.size(size_dim) * self.scale_factor))
                    + sum(self._padding)
                    - sum(self._padding_const_size)
                )
                h = nn.functional.interpolate(
                    h, size=hs_size, mode=self.skip_interp_mode
                )
            hs = self.skip_connection(h)
            y = 0.5 * hs + 0.5 * hb
        else:
            y = hb
        return y

    def init_parameters(self):
        r"""
        Initializes the values of trainable parameters.
        """
        if hasattr(self.block, "activation"):
            gain = _get_gain(self.block.activation)
        else:
            gain = _get_gain(None)
        _set_init_parameters(self.block.layer_0, gain)
        if hasattr(self.block, "layer_1"):
            _set_init_parameters(self.block.layer_1, gain)
        if hasattr(self.block, "layer_2"):
            _set_init_parameters(self.block.layer_2, gain)
        if self.skip_connection is not None and not isinstance(
            self.skip_connection, nn.Identity
        ):
            _set_init_parameters(self.skip_connection)


class DownsampleBlock(MultiLevelBlock):
    """
    Downsampling based on the universal-design multi-level block.

    Args:
      input_channels   Channels in the inputs.
      output_channels  Channels in the outputs.
    """

    def __init__(
        self,
        input_channels,
        kernel_size,
        scale_factor=0.5,
        skip_connection=True,
        interp_mode="nearest-exact",
        **mlb_kwargs,
    ):
        super().__init__(
            input_channels,
            kernel_size,
            scale_factor=scale_factor,
            skip_connection=skip_connection,
            interp_mode=interp_mode,
            logger=logging.getLogger("dlkit.nets.conv1d.DownsampleBlock"),
            **mlb_kwargs,
        )


class UpsampleBlock(MultiLevelBlock):
    """
    Upsampling based on the universal-design multi-level block.

    Args:
      input_channels   Channels in the inputs.
      output_channels  Channels in the outputs.
    """

    def __init__(
        self,
        input_channels,
        kernel_size,
        scale_factor=2,
        skip_connection=True,
        interp_mode="nearest-exact",
        **mlb_kwargs,
    ):
        super().__init__(
            input_channels,
            kernel_size,
            scale_factor=scale_factor,
            skip_connection=skip_connection,
            interp_mode=interp_mode,
            logger=logging.getLogger("dlkit.nets.conv1d.UpsampleBlock"),
            **mlb_kwargs,
        )


class LevelBlock(MultiLevelBlock):
    """
    Residual block based on the universal-design multi-level block.

    Args:
      input_channels   Channels in the inputs.
      output_channels  Channels in the outputs.
    """

    def __init__(
        self,
        input_channels,
        kernel_size,
        skip_connection=True,
        **mlb_kwargs,
    ):
        super().__init__(
            input_channels,
            kernel_size,
            skip_connection=skip_connection,
            logger=logging.getLogger("dlkit.nets.conv1d.LevelBlock"),
            **mlb_kwargs,
        )


# --------------------------------------
# Utility Functions
# --------------------------------------


def _get_conv1d_size(in_length, kernel, stride=1, padding=0, dilation=1):
    return int( (in_length + 2*padding - dilation*(kernel- 1) - 1) / stride + 1 )


def _get_gain(activation):
    r"""
    Calculates the gain to be used as an argument for initializing parameter values.

    Args:
        activation: Object of activation function or None
    """
    if activation is not None:
        activation_name = type(activation).__name__.lower()
        if activation_name in ["silu", "gelu"]:
            activation_name = "relu"
        gain = nn.init.calculate_gain(activation_name)
    else:
        gain = nn.init.calculate_gain("conv1d")
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


def test_ConvNet():
    print("---------------------------------------^")
    input_channels = 1
    input_size = 16
    hidden_conv_layers_channels_mult = [2, 4, 8]
    hidden_dense_input_size = (
        16 - 2 * len(hidden_conv_layers_channels_mult)
    ) * hidden_conv_layers_channels_mult[-1]
    net = ConvNet(
        input_channels,
        output_size=2,
        hidden_conv_layers_channels_mult=hidden_conv_layers_channels_mult,
        hidden_dense_input_size=hidden_dense_input_size,
        hidden_dense_layers_sizes=[32, 32],
    )
    print(net)

    print("Test 1:")
    x = torch.ones((1, input_channels, input_size))
    y = net(x)
    print("- input  x =", x, sep="\n")
    print("- output y =", y, sep="\n")
    print("---------------------------------------$")


def test_ConvResNet():
    """Test ConvResNet architecture."""
    print("---------------------------------------^")
    batch_size = 4
    input_channels = 1
    input_length = 64
    output_size = 10
    channels_mult = [4, 8]

    # test basic configuration without MLPResNet
    print("Test 1: basic configuration (conv layers only)")
    net = ConvResNet(
        input_channels=input_channels,
        conv_resnet_params={
            "channels_mult": channels_mult,
            "kernels": [3, 3],
            "activation": nn.ReLU(),
        },
    )
    print(net)
    x = torch.randn(batch_size, input_channels, input_length)
    # calculate size after conv layers with 2 downsampling layers (scale_factor=2)
    # and output channels = input_channels * channels_mult[-1]
    expected_out_size = (
        batch_size,
        input_channels * channels_mult[-1],
        input_length // 4,
    )
    expected_flat_size = (batch_size, expected_out_size[1] * expected_out_size[2])
    print(f"Expected output size: {expected_out_size}")
    y = net(x)
    print(f"Output size: {y.size()}")
    assert (
        y.size() == expected_out_size
    ), f"Expected {expected_out_size}, got {y.size()}"

    # test with MLPResNet
    print("\nTest 2: with MLPResNet")
    net = ConvResNet(
        input_channels=input_channels,
        conv_resnet_params={
            "channels_mult": [4, 8],
            "kernels": [3, 3],
            "activation": nn.ReLU(),
        },
        mlp_resnet_params={
            "input_size": expected_flat_size[1],
            "output_size": output_size,
            "residual_blocks_sizes": [(16, 16, 64, 16)],
        },
    )
    print(net)
    x = torch.randn(batch_size, input_channels, input_length)
    expected_out_size = (batch_size, output_size)
    print(f"Expected output size: {expected_out_size}")
    y = net(x)
    print(f"Output size: {y.size()}")
    assert (
        y.size() == expected_out_size
    ), f"Expected {expected_out_size}, got {y.size()}"

    # test with hidden inputs to MLPResNet
    print("\nTest 3: with hidden inputs")
    hidden_input_size = 8
    net = ConvResNet(
        input_channels=input_channels,
        conv_resnet_params={
            "channels_mult": [4, 8],
            "kernels": [3, 3],
            "activation": nn.ReLU(),
        },
        mlp_resnet_params={
            "input_size": expected_flat_size[1],
            "output_size": output_size,
            "residual_blocks_sizes": [
                (16, 16, 64, 16),
                (16 + hidden_input_size, 16, 64, 16),
            ],
        },
    )
    print(net)
    x = torch.randn(batch_size, input_channels, input_length)
    h1 = torch.randn(batch_size, 1, hidden_input_size)  # input to 2nd residual block
    print(f"Expected output size: {expected_out_size}")
    y = net(x, h1=h1)
    print(f"Output size: {y.size()}")
    assert (
        y.size() == expected_out_size
    ), f"Expected {expected_out_size}, got {y.size()}"

    print("---------------------------------------$")


def test_MultiLevelBlock():
    """Test MultiLevelBlock module."""
    print("---------------------------------------^")
    batch_size = 4
    input_channels = 16
    input_length = 64

    x = torch.randn(batch_size, input_channels, input_length)

    #
    # Test Block
    #

    expected_out_size = (batch_size, input_channels, input_length)

    # test basic configuration without skip connection
    print("Test 1: basic configuration (no skip connection)")
    net = MultiLevelBlock(
        input_channels=input_channels,
        kernel_size=3,
    )
    print(net)
    print(f"Expected output size: {expected_out_size}")
    y = net(x)
    print(f"Output size: {y.size()}")
    assert (
        y.size() == expected_out_size
    ), f"Expected {expected_out_size}, got {y.size()}"

    # test single-layer configuration
    print("\nTest 2: single-layer configuration")
    net = MultiLevelBlock(
        input_channels=input_channels,
        kernel_size=3,
        normalization=False,
        normalization_layer_channels=False,
        activation_layer_channels=False,
    )
    print(net)
    print(f"Expected output size: {expected_out_size}")
    y = net(x)
    print(f"Output size: {y.size()}")
    assert (
        y.size() == expected_out_size
    ), f"Expected {expected_out_size}, got {y.size()}"

    # test with custom channels and normalization
    print("\nTest 3: with custom channels and normalization")
    output_channels = 32
    normalization_layer_channels = 8
    activation_layer_channels = 64
    net = MultiLevelBlock(
        input_channels=input_channels,
        kernel_size=3,
        normalization=nn.GroupNorm(2, normalization_layer_channels),
        normalization_layer_channels=normalization_layer_channels,
        activation=nn.SiLU(),
        activation_layer_channels=activation_layer_channels,
        output_channels=output_channels,
    )
    expected_out_size = (batch_size, output_channels, input_length)
    print(net)
    print(f"Expected output size: {expected_out_size}")
    y = net(x)
    print(f"Output size: {y.size()}")
    assert (
        y.size() == expected_out_size
    ), f"Expected {expected_out_size}, got {y.size()}"

    # test with skip connection
    print("\nTest 4: with skip connection")
    output_channels = 32
    net = MultiLevelBlock(
        input_channels=input_channels,
        kernel_size=3,
        output_channels=output_channels,
        skip_connection=True,
    )
    expected_out_size = (batch_size, output_channels, input_length)
    print(net)
    print(f"Expected output size: {expected_out_size}")
    y = net(x)
    print(f"Output size: {y.size()}")
    assert (
        y.size() == expected_out_size
    ), f"Expected {expected_out_size}, got {y.size()}"

    # test with dropout
    print("\nTest 5: with dropout")
    net = MultiLevelBlock(
        input_channels=input_channels,
        kernel_size=3,
        dropout=nn.Dropout(0.1, inplace=False),
        skip_connection=True,
    )
    expected_out_size = (batch_size, input_channels, input_length)
    print(net)
    print(f"Expected output size: {expected_out_size}")
    y = net(x)
    print(f"Output size: {y.size()}")
    assert (
        y.size() == expected_out_size
    ), f"Expected {expected_out_size}, got {y.size()}"

    #
    # Test Downsample
    #

    expected_out_size = (batch_size, input_channels, input_length // 2)

    # test single-layer downsample
    print("\nTest 6: single-layer downsample (no skip connection)")
    net = MultiLevelBlock(
        input_channels=input_channels,
        kernel_size=3,
        scale_factor=0.5,
        normalization=False,
        normalization_layer_channels=False,
        activation_layer_channels=False,
    )
    print(net)
    print(f"Expected output size: {expected_out_size}")
    y = net(x)
    print(f"Output size: {y.size()}")
    assert (
        y.size() == expected_out_size
    ), f"Expected {expected_out_size}, got {y.size()}"

    # test downsample with skip connection
    print("\nTest 7: downsample with skip connection")
    net = MultiLevelBlock(
        input_channels=input_channels,
        kernel_size=3,
        scale_factor=0.5,
        skip_connection=True,
    )
    print(net)
    print(f"Expected output size: {expected_out_size}")
    y = net(x)
    print(f"Output size: {y.size()}")
    assert (
        y.size() == expected_out_size
    ), f"Expected {expected_out_size}, got {y.size()}"

    #
    # Test Upsample
    #

    expected_out_size = (batch_size, input_channels, input_length * 2)

    # test single-layer upsample
    print("\nTest 8: single-layer upsample (no skip connection)")
    net = MultiLevelBlock(
        input_channels=input_channels,
        kernel_size=3,
        scale_factor=2.0,
        normalization=False,
        normalization_layer_channels=False,
        activation_layer_channels=False,
    )
    print(net)
    print(f"Expected output size: {expected_out_size}")
    y = net(x)
    print(f"Output size: {y.size()}")
    assert (
        y.size() == expected_out_size
    ), f"Expected {expected_out_size}, got {y.size()}"

    # test upsample with skip connection
    print("\nTest 9: upsample with skip connection")
    net = MultiLevelBlock(
        input_channels=input_channels,
        kernel_size=3,
        scale_factor=2.0,
        skip_connection=True,
    )
    print(net)
    print(f"Expected output size: {expected_out_size}")
    y = net(x)
    print(f"Output size: {y.size()}")
    assert (
        y.size() == expected_out_size
    ), f"Expected {expected_out_size}, got {y.size()}"

    print("---------------------------------------$")


if __name__ == "__main__":
    r"""Runs tests."""
    test_ConvNet()
    print("\n")
    test_MultiLevelBlock()
    print("\n")
    test_ConvResNet()
