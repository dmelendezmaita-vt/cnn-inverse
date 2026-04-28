"""
Create neural networks.
"""

import math
import copy
import torch
import torch.nn as nn
import numpy as np

from dlkit.nets.mlp import (
        MLPNet_MultIn,
        MLPResNet
)
from dlkit.nets.conv1d import (
    ConvNet,
    ConvResNet
)
from dlkit.nets.efficientnet import EfficientNet1D
from dlkit.nets.transformer1d import (
        TransformerNet,
        ChannelWiseTransformerNet
)
from dlkit.nets.autoencoder import Autoencoder
from dlkit.nets.unet import UNet1d_2021 as UNet
from dlkit.nets.unet import EncoderNet1d_2021 as EncoderConvNet
from dlkit.nets.unet import DecoderNet1d_2021 as DecoderConvNet

from utils import NetworkType

# --------------------------------------
# Neural Networks for Inverse Maps
# --------------------------------------


def _make_pool(pool_type, kernel, stride, padding):
    if pool_type is None:
        return None
    if isinstance(pool_type, str):
        pool_type_cf = pool_type.casefold()
        if pool_type_cf == "avg":
            return nn.AvgPool1d(kernel_size=kernel, stride=stride, padding=padding)
        if pool_type_cf == "max":
            return nn.MaxPool1d(kernel_size=kernel, stride=stride, padding=padding)
    raise ValueError(f"Unknown conv_pool_type: {pool_type!r}")


class ConvMultiHeadNet(nn.Module):
    def __init__(
        self,
        input_channels,
        input_size,
        output_size,
        *,
        conv_layer_sizes,
        dense_layer_sizes,
        activation_fn,
        conv_layer_kernel=3,
        conv_layer_stride=2,
        conv_layer_padding=0,
        conv_pool_type=None,
        conv_pool_kernel=2,
        conv_pool_stride=2,
        conv_pool_padding=0,
        head_strategy="shared",
        head_hidden_sizes=None,
        head_groups=None,
        use_dropout=False,
    ):
        super().__init__()
        self.output_size = int(output_size)
        self.head_strategy = str(head_strategy).strip().casefold()
        self.activation_template = activation_fn
        self.dropout = nn.Dropout(p=0.1) if use_dropout else None

        self.conv_blocks = nn.ModuleList()
        in_channels = int(input_channels)
        feature_size = int(input_size)
        pool = _make_pool(conv_pool_type, conv_pool_kernel, conv_pool_stride, conv_pool_padding)
        for out_mult in conv_layer_sizes:
            out_channels = int(input_channels) * int(out_mult)
            layers = [
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=int(conv_layer_kernel),
                    stride=int(conv_layer_stride),
                    padding=int(conv_layer_padding),
                ),
                copy.deepcopy(self.activation_template),
            ]
            if use_dropout:
                layers.append(nn.Dropout(p=0.1))
            if pool is not None:
                layers.append(_make_pool(conv_pool_type, conv_pool_kernel, conv_pool_stride, conv_pool_padding))
            self.conv_blocks.append(nn.Sequential(*layers))
            feature_size = _get_conv1d_size(feature_size, int(conv_layer_kernel), int(conv_layer_stride), int(conv_layer_padding))
            if pool is not None:
                feature_size = _get_conv1d_size(feature_size, int(conv_pool_kernel), int(conv_pool_stride), int(conv_pool_padding))
            in_channels = out_channels

        shared_layers = []
        shared_in = in_channels * feature_size
        for hidden in dense_layer_sizes:
            shared_layers.append(nn.Linear(shared_in, int(hidden)))
            shared_layers.append(copy.deepcopy(self.activation_template))
            if use_dropout:
                shared_layers.append(nn.Dropout(p=0.1))
            shared_in = int(hidden)
        self.shared_mlp = nn.Sequential(*shared_layers) if shared_layers else nn.Identity()
        self.shared_dim = shared_in

        if head_hidden_sizes is None:
            head_hidden_sizes = []
        self.head_hidden_sizes = [int(v) for v in head_hidden_sizes]

        if self.head_strategy == "shared":
            self.shared_head = self._make_head(self.output_size)
            self.heads = None
            self.head_groups = None
        elif self.head_strategy == "per_target":
            self.heads = nn.ModuleList([self._make_head(1) for _ in range(self.output_size)])
            self.shared_head = None
            self.head_groups = None
        elif self.head_strategy == "grouped":
            if not head_groups:
                raise ValueError("ConvMultiHeadNet grouped strategy requires head_groups")
            self.head_groups = [list(map(int, grp)) for grp in head_groups]
            self.heads = nn.ModuleList([self._make_head(len(grp)) for grp in self.head_groups])
            self.shared_head = None
        else:
            raise ValueError(f"Unknown head_strategy: {head_strategy!r}")

    def _make_head(self, out_dim: int) -> nn.Module:
        layers = []
        in_dim = self.shared_dim
        for hidden in self.head_hidden_sizes:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(copy.deepcopy(self.activation_template))
            if self.dropout is not None:
                layers.append(nn.Dropout(p=0.1))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, out_dim))
        return nn.Sequential(*layers)

    def forward(self, x):
        for block in self.conv_blocks:
            x = block(x)
        x = torch.flatten(x, start_dim=1)
        x = self.shared_mlp(x)
        if self.head_strategy == "shared":
            return self.shared_head(x)
        if self.head_strategy == "per_target":
            return torch.cat([head(x) for head in self.heads], dim=1)
        out = x.new_zeros((x.shape[0], self.output_size))
        for head, group in zip(self.heads, self.head_groups):
            out[:, group] = head(x)
        return out

def _get_activation(name):
    if 'relu' == name.casefold():
        return nn.ReLU()
    elif 'gelu' == name.casefold():
        return nn.GELU()
    elif 'silu' == name.casefold() or 'swish' == name.casefold():
        return nn.SiLU()
    else:
        raise ValueError('Unknown name for activation function: '+name)

def _get_conv1d_size(in_length, kernel, stride=1, padding=0, dilation=1):
    return int( (in_length + 2*padding - dilation*(kernel- 1) - 1) / stride + 1 )

def _create_MLPNet(input_channels, input_size, output_size, net_params, logger):
    return MLPNet_MultIn(
            input_channels*input_size,
            output_size,
            hidden_layers_sizes      = net_params['dense_layer_sizes'],
            hidden_layers_activation = _get_activation(net_params['activation_fn']),
            use_dropout              = net_params.get('dropout', False),
            output_layer_activation  = None
    )

def _create_MLPResNet(input_channels, input_size, output_size, net_params, logger,
                      hidden_input_size=0):
    activation_fn = _get_activation(net_params['activation_fn'])
    embed_size    = net_params.get('embedding_size', 1)
    flattened_input_size = (input_channels * input_size,
                            net_params['residual_blocks_sizes'][0][0])
    # configure hidden inputs to residual blocks
    if hidden_input_size:
        residual_blocks_sizes = list(net_params['residual_blocks_sizes'])
        if isinstance(hidden_input_size, int):
            hidden_input_size = [hidden_input_size] * len(residual_blocks_sizes)
        assert len(hidden_input_size) == len(residual_blocks_sizes)
        for i in range(len(residual_blocks_sizes)):
            residual_blocks_sizes[i][0] += hidden_input_size[i]
    else:
        residual_blocks_sizes = net_params['residual_blocks_sizes']
    # create net
    return MLPResNet(
            flattened_input_size,
            output_size,
            embedding_size                   = embed_size,
            attention_blocks_n_heads         = net_params.get('attention_layers_n_heads'),
            attention_blocks_activation_size = embed_size * 4,
            attention_blocks_activation      = activation_fn,
            residual_blocks_sizes            = net_params['residual_blocks_sizes'],
            residual_blocks_activation       = activation_fn,
            use_dropout                      = net_params.get('dropout', False),
            output_layer_activation          = None
    )

def _create_convNet(input_channels, input_size, output_size, net_params, logger):
    activation_fn = _get_activation(net_params['activation_fn'])
    use_dropout = net_params.get('dropout', False)
    kernel = net_params.get("conv_layer_kernel", 3)
    stride = net_params.get("conv_layer_stride", 2)
    padding = net_params.get("conv_layer_padding", 0)
    pool_type = net_params.get("conv_pool_type")
    if isinstance(pool_type, str) and not pool_type.strip():
        pool_type = None
    pool_kernel = net_params.get("conv_pool_kernel", 2)
    pool_stride = net_params.get("conv_pool_stride", 2)
    pool_padding = net_params.get("conv_pool_padding", 0)
    n_conv_layers = len(net_params['conv_layer_sizes'])
    # set parameters of convolution layers
    hidden_conv_layers_kernels = n_conv_layers * [kernel]
    hidden_conv_layers_kwargs = {'stride': stride, 'padding': padding}
    hidden_pool_layers_kernels = None
    hidden_pool_layers_kwargs = {}
    if pool_type is not None:
        hidden_pool_layers_kernels = n_conv_layers * [pool_kernel]
        hidden_pool_layers_kwargs = {'stride': pool_stride, 'padding': pool_padding}
    # calculate length of features after convolutional layers
    n_channels = input_channels * net_params['conv_layer_sizes'][-1]
    n_features = input_size
    for _ in range(n_conv_layers):
        n_features = _get_conv1d_size(n_features, kernel, stride, padding)
        if hidden_pool_layers_kernels is not None:
            n_features = _get_conv1d_size(n_features, pool_kernel, pool_stride, pool_padding)
    flattened_input_size = n_channels * n_features
    # create net
    return ConvNet(
        # convolutional layers
        input_channels,
        hidden_conv_layers_channels_mult = net_params['conv_layer_sizes'],
        hidden_conv_layers_kernels       = hidden_conv_layers_kernels,
        hidden_conv_layers_activation    = activation_fn,
        hidden_conv_layers_kwargs        = hidden_conv_layers_kwargs,
        hidden_pool_layers_type          = pool_type,
        hidden_pool_layers_kernels       = hidden_pool_layers_kernels,
        hidden_pool_layers_kwargs        = hidden_pool_layers_kwargs,
        # dense layers
        hidden_dense_input_size          = flattened_input_size,
        hidden_dense_layers_sizes        = net_params['dense_layer_sizes'],
        hidden_dense_layers_activation   = activation_fn,
        # output layer
        output_size                      = output_size,
        output_layer_activation          = None,
        # other
        use_dropout                      = use_dropout,
    )


def _create_convMultiHeadNet(input_channels, input_size, output_size, net_params, logger):
    activation_fn = _get_activation(net_params['activation_fn'])
    use_dropout = net_params.get('dropout', False)
    return ConvMultiHeadNet(
        input_channels,
        input_size,
        output_size,
        conv_layer_sizes=net_params['conv_layer_sizes'],
        dense_layer_sizes=net_params['dense_layer_sizes'],
        activation_fn=activation_fn,
        conv_layer_kernel=net_params.get("conv_layer_kernel", 3),
        conv_layer_stride=net_params.get("conv_layer_stride", 2),
        conv_layer_padding=net_params.get("conv_layer_padding", 0),
        conv_pool_type=net_params.get("conv_pool_type"),
        conv_pool_kernel=net_params.get("conv_pool_kernel", 2),
        conv_pool_stride=net_params.get("conv_pool_stride", 2),
        conv_pool_padding=net_params.get("conv_pool_padding", 0),
        head_strategy=net_params.get("target_head_strategy", "shared"),
        head_hidden_sizes=net_params.get("head_dense_layer_sizes", []),
        head_groups=net_params.get("target_head_groups"),
        use_dropout=use_dropout,
    )

def _create_convResNet(input_channels, input_size, output_size, net_params, logger,
                       mlp_block_hidden_input_size=0):
    activation_fn = _get_activation(net_params['activation_fn'])
    use_dropout = net_params.get('dropout', False)
    kernel = net_params.get("conv_layer_kernel", 3)
    stride = net_params.get("conv_layer_stride", 2)
    padding = net_params.get("conv_layer_padding", 1)
    padding_mode = net_params.get("conv_layer_padding_mode", "replicate")
    n_conv_layers = len(net_params['conv_layer_sizes'])
    # set parameters of convolution block
    conv_resnet_params = {
        "channels_mult": net_params['conv_layer_sizes'],
        "kernels": n_conv_layers * [kernel],
        "activation": activation_fn,
        "use_dropout": use_dropout,
        "mlb_kwargs": {
            'stride': stride,
            'padding': padding,
            'padding_mode': padding_mode,
        },
    }
    # calculate length of features after convolutional layers
    n_channels = input_channels * net_params['conv_layer_sizes'][-1]
    n_features = input_size
    for _ in range(n_conv_layers):
        n_features = _get_conv1d_size(n_features, kernel, stride, padding)
    flattened_input_size = (
        n_channels * n_features, net_params['residual_blocks_sizes'][0][0]
    )
    # configure hidden inputs to residual blocks
    if mlp_block_hidden_input_size:
        residual_blocks_sizes = list(net_params['residual_blocks_sizes'])
        if isinstance(mlp_block_hidden_input_size, int):
            mlp_block_hidden_input_size = [mlp_block_hidden_input_size] * len(residual_blocks_sizes)
        assert len(mlp_block_hidden_input_size) == len(residual_blocks_sizes)
        for i in range(len(residual_blocks_sizes)):
            residual_blocks_sizes[i][0] += mlp_block_hidden_input_size[i]
    else:
        residual_blocks_sizes = net_params['residual_blocks_sizes']
    # set parameters of MLP block
    mlp_resnet_params = {
        "input_size": flattened_input_size,
        "output_size": output_size,
        "residual_blocks_sizes": residual_blocks_sizes,
        "residual_blocks_activation": activation_fn,
        "use_dropout": use_dropout,
        "output_layer_activation": None,
    }
    # create net
    return ConvResNet(
            input_channels,
            conv_resnet_params=conv_resnet_params,
            mlp_resnet_params=mlp_resnet_params,
    )

def _create_efficientNet(input_channels, input_size, output_size, net_params, logger):
    use_dropout = net_params.get('dropout', False)
    return EfficientNet1D(
            input_channels  = input_channels,
            input_length    = input_size,
            num_classes     = output_size,
            dropout_connect = use_dropout if use_dropout else 0.0,
            dropout_head    = use_dropout if use_dropout else 0.0
    )

def _create_transformerNet(input_channels, input_size, output_size, net_params, logger):
    patch_size   = net_params.get('patch_size', input_size//10)
    embed_size   = net_params.get('embedding_size')
    attn_n_heads = net_params.get('attention_layers_n_heads')
    use_dropout  = net_params.get('dropout', False)
    if 1 == input_channels:
        return TransformerNet(
                input_seq_size = input_size,
                output_size    = output_size,
                patch_size     = patch_size,
                embedding_size = embed_size,
                attn_n_heads   = attn_n_heads,
                dropout        = use_dropout if use_dropout else 0.0
        )
    else:
        return ChannelWiseTransformerNet(
                input_channels = input_channels,
                input_seq_size = input_size,
                output_size    = output_size,
                patch_size     = patch_size,
                embedding_size = embed_size,
                attn_n_heads   = attn_n_heads,
                dropout        = use_dropout if use_dropout else 0.0
        )

def create_network(params, logger):
    # get network options
    net_params = params['net']
    net_type   = NetworkType.get_from_name(net_params['type'])
    logger.info(f"Network type: {net_params['type']}, key: {net_type}")
    # set input and output sizes
    assert 2 == len(params['data']['num_features'])
    input_channels, input_size = params['data']['num_features']
    output_size = params['data']['num_targets']
    # create network
    if NetworkType.MLPNET == net_type:
        net = _create_MLPNet(input_channels, input_size, output_size, net_params, logger)
    elif NetworkType.MLPRESNET == net_type:
        net = _create_MLPResNet(input_channels, input_size, output_size, net_params, logger)
    elif NetworkType.CONVNET == net_type:
        net = _create_convNet(input_channels, input_size, output_size, net_params, logger)
    elif NetworkType.CONVMULTIHEAD == net_type:
        net = _create_convMultiHeadNet(input_channels, input_size, output_size, net_params, logger)
    elif NetworkType.CONVRESNET == net_type:
        net = _create_convResNet(input_channels, input_size, output_size, net_params, logger)
    elif NetworkType.EFFICIENTNET == net_type:
        net = _create_efficientNet(input_channels, input_size, output_size, net_params, logger)
    elif NetworkType.TRANSFORMERNET == net_type:
        net = _create_transformerNet(input_channels, input_size, output_size, net_params, logger)
    else:
        raise NotImplementedError(f"Type {net_type} is not implemented")
    if net_params.get("learn_task_uncertainty", False):
        net.log_task_vars = nn.Parameter(torch.zeros(output_size, dtype=torch.float32))
    # return network
    return net

# --------------------------------------
# Autoencoder Networks
# --------------------------------------

def create_enc_dec(params, logger):
    e_net_params    = params['e_net']
    d_net_params    = params['d_net']
    e_net_type      = NetworkType.get_from_name(e_net_params['type'])
    d_net_type      = NetworkType.get_from_name(d_net_params['type'])
    logger.info(f"Encoder type: {e_net_params['type']}, key: {e_net_type}")
    logger.info(f"Decoder type: {d_net_params['type']}, key: {d_net_type}")
    input_size  = math.prod(params['data']['num_features'])
    latent_size = params['data']['latent_size']
    output_size = input_size
    input_channels  = params['data']['num_features'][0]
    latent_channels = params['data']['num_features'][0]
    output_channels = params['data']['num_features'][0]
    # create encoder network
    if NetworkType.MLPNET == e_net_type:
        e_net = MLPNet(
                input_size,
                latent_size,
                hidden_layers_sizes      = e_net_params['dense_layer_sizes'],
                hidden_layers_activation = _get_activation(e_net_params['activation_fn']),
                use_dropout              = e_net_params['dropout']
        )
    elif NetworkType.MLPRESNET == e_net_type:
        e_net = MLPResNet(
                input_size,
                latent_size,
                residual_blocks_sizes      = e_net_params['residual_blocks_sizes'],
                residual_blocks_activation = _get_activation(e_net_params['activation_fn']),
                use_dropout                = e_net_params['dropout'],
        )
    elif NetworkType.CONVNET == e_net_type:
        e_net = EncoderConvNet(
                input_channels,
                latent_channels,
                internal_channels = e_net_params['conv_internal_channels'],
                channel_mult      = e_net_params['conv_channel_mult']
        )
    else:
        raise NotImplementedError()
    # create decoder network
    if NetworkType.MLPNET == d_net_type:
        d_net = MLPNet(
                latent_size,
                output_size,
                hidden_layers_sizes      = d_net_params['dense_layer_sizes'],
                hidden_layers_activation = _get_activation(d_net_params['activation_fn']),
                use_dropout              = d_net_params['dropout'],
                output_layer_activation  = None
        )
    elif NetworkType.MLPRESNET == d_net_type:
        d_net = MLPResNet(
                latent_size,
                output_size,
                residual_blocks_sizes      = d_net_params['residual_blocks_sizes'],
                residual_blocks_activation = _get_activation(d_net_params['activation_fn']),
                use_dropout                = d_net_params['dropout']
        )
    elif NetworkType.CONVNET == d_net_type:
        d_net = DecoderConvNet(
                latent_channels,
                output_channels,
                internal_channels = d_net_params['conv_internal_channels'],
                channel_mult      = d_net_params['conv_channel_mult']
        )
    else:
        raise NotImplementedError()
    # return networks
    return e_net, d_net

def create_ae(params, logger):
    e_net, d_net = create_enc_dec(params, logger)
    def _output_transf(y):
        return y.reshape(-1, *params['data']['num_features'])
    return Autoencoder(e_net, d_net, output_layer_transformation=_output_transf)

# --------------------------------------
# UNet
# --------------------------------------

def create_unet(params, logger):
    return UNet(1, 1,
                internal_channels = params['net']['conv_internal_channels'],
                channel_mult      = params['net']['conv_channel_mult'])

# --------------------------------------
# Generative Adversarial Networks
# --------------------------------------

class GNet(nn.Module):
    def __init__(self, input_channels, input_size, latent_size, targets_size, net_params, logger):
        super().__init__()
        net_type = NetworkType.get_from_name(net_params['type'])
        logger.info(f"Generator network type:     {net_params['type']}, key: {net_type}")
        self.hidden_input_size = None
        # create network
        if NetworkType.MLPNET == net_type:
            self.net = _create_MLPNet(input_channels, input_size+latent_size, targets_size, net_params, logger)
        elif NetworkType.MLPRESNET == net_type:
            self.hidden_input_size = [0] * len(net_params["residual_blocks_sizes"])
            self.hidden_input_size[0] = latent_size
            # latent_size can also be propagated to subsequent layers if needed
            self.net = _create_MLPResNet(input_channels, input_size, targets_size, net_params, logger,
                                         hidden_input_size=self.hidden_input_size)
        # elif NetworkType.CONVNET == net_type:
        #     self.net = _create_convNet(input_channels+latent_size, input_size, targets_size, net_params, logger)
        elif NetworkType.CONVRESNET == net_type:
            self.hidden_input_size = [0] * len(net_params["residual_blocks_sizes"])
            self.hidden_input_size[0] = latent_size
            # latent_size can also be propagated to subsequent layers if needed
            self.net = _create_convResNet(input_channels, input_size, targets_size, net_params, logger,
                                          mlp_block_hidden_input_size=self.hidden_input_size)
        # elif NetworkType.EFFICIENTNET == net_type:
        #     self.net = _create_efficientNet(input_channels+latent_size, input_size, targets_size, net_params, logger)
        # elif NetworkType.TRANSFORMERNET == net_type:
        #     self.net = _create_transformerNet(input_channels+latent_size, input_size, targets_size, net_params, logger)
        else:
            raise NotImplementedError(f"Type {net_type} is not implemented")

    def forward(self, y, z):
        n_replica = z.size(0) if z.size(0) != y.size(0) else 0
        if 0 == n_replica:
            if self.hidden_input_size:
                g_out = self.net(y, h0=z)
            else:
                g_out = self.net(y, z)
        elif 1 == n_replica:
            if self.hidden_input_size:
                g_out = self.net(y, h0=z[0])
            else:
                g_out = self.net(y, z[0])
        elif 1 < n_replica:
            if self.hidden_input_size:
                g_out = torch.vmap( lambda z_: self.net(y, h0=z_) )(z)
            else:
                g_out = torch.vmap( lambda z_: self.net(y, z_) )(z)
        else:
            raise NotImplementedError(f"{y.size()=}, {z.size()=}")
        return g_out


class DNet(nn.Module):
    def __init__(self, input_channels, input_size, targets_size, net_params, logger):
        super().__init__()
        net_type = NetworkType.get_from_name(net_params['type'])
        logger.info(f"Discriminator network type: {net_params['type']}, key: {net_type}")
        self.hidden_input_size = None
        # create network
        if NetworkType.MLPNET == net_type:
            self.net = _create_MLPNet(input_channels, input_size+targets_size, 1, net_params, logger)
        elif NetworkType.MLPRESNET == net_type:
            self.hidden_input_size = [0] * len(net_params["residual_blocks_sizes"])
            self.hidden_input_size[0] = targets_size
            self.net = _create_MLPResNet(input_channels, input_size, 1, net_params, logger,
                                         hidden_input_size=self.hidden_input_size)
        # elif NetworkType.CONVNET == net_type:
        #     self.net = _create_convNet(input_channels+targets_size, input_size, 1, net_params, logger)
        elif NetworkType.CONVRESNET == net_type:
            self.hidden_input_size = [0] * len(net_params["residual_blocks_sizes"])
            self.hidden_input_size[0] = targets_size
            self.net = _create_convResNet(input_channels, input_size, 1, net_params, logger,
                                          mlp_block_hidden_input_size=self.hidden_input_size)
        # elif NetworkType.EFFICIENTNET == net_type:
        #     self.net = _create_efficientNet(input_channels+targets_size, input_size, 1, net_params, logger)
        # elif NetworkType.TRANSFORMERNET == net_type:
        #     self.net = _create_transformerNet(input_channels+targets_size, input_size, 1, net_params, logger)
        else:
            raise NotImplementedError(f"Type {net_type} is not implemented")

    def forward(self, x, y):
        n_replica = x.size(0) if x.size(0) != y.size(0) else 0
        if 0 == n_replica:
            if self.hidden_input_size:
                d_out = self.net(y, h0=x)
            else:
                d_out = self.net(x, y)
        elif 1 == n_replica:
            if self.hidden_input_size:
                d_out = self.net(y, h0=x[0])
            else:
                d_out = self.net(x[0], y)
        elif 1 < n_replica:
            if self.hidden_input_size:
                d_out = torch.vmap( lambda x_: self.net(y, h0=x_) )(x)
            else:
                d_out = torch.vmap( lambda x_: self.net(x_, y) )(x)
            d_out = d_out.flatten(start_dim=0, end_dim=1)  # return flattened tensor without dim=0
        else:
            raise NotImplementedError(f"{x.size()=}, {y.size()=}")
        return d_out


def create_gan(params, logger):
    # get network options
    g_net_params = params['g_net']
    d_net_params = params['d_net']
    # set input and output sizes
    assert 2 == len(params['data']['num_features'])
    input_channels, input_size = params['data']['num_features']
    output_size = params['data']['num_targets']
    latent_size = params['data']['latent_size']
    # create generator & discriminator networks
    g_net = GNet(input_channels, input_size, latent_size, output_size, g_net_params, logger)
    d_net = DNet(input_channels, input_size, output_size, d_net_params, logger)
    # return networks
    return g_net, d_net

### OLD:

#class DenseNet(nn.Module):
#    def __init__(self, params):
#        super().__init__()
#        model_params = params['model']
#        activation_fn = _get_activation(model_params['activation_fn'])
#        self.flatten = nn.Flatten()  # flattens the input tensor
#
#        # create linear layers
#        self.hidden_layers = nn.ModuleList()  # list of hidden layers
#        in_features = model_params['num_features'][1]
#        for out_features in model_params['dense_layer_sizes']:
#            self.hidden_layers.append(nn.Sequential(
#                nn.Linear(in_features, out_features),
#                activation_fn,
#                nn.Dropout(p=model_params['dropout']) if model_params['dropout'] else nn.Identity()
#            ).apply(self.init_weights))
#            in_features = out_features
#
#        # set output layer
#        self.output_layer = nn.Sequential(
#            nn.Linear(in_features, model_params['num_targets']),
#            nn.Sigmoid()
#        ).apply(self.init_weights)
#
#    def init_weights(self,m):
#        if isinstance(m, nn.Linear):
#            nn.init.xavier_uniform_(m.weight)
#
#    def forward(self, x):
#        x = self.flatten(x)
#        for layer in self.hidden_layers:
#            x = layer(x)
#        x = self.output_layer(x)
#        return x

#class ConvNet(nn.Module):
#    def __init__(self, params):
#        super().__init__()
#        model_params = params['model']
#        activation_fn = _get_activation(model_params['activation_fn'])
#        initializer = nn.init.xavier_uniform_  # use Xavier initialization
#        conv_kernel = 3
#        conv_stride = 2
#        conv_padding = 0
#        pool_fn = nn.AvgPool1d  # AvgPool1d or MaxPool1d
#        pool_kernel = 2
#        pool_stride = 2
#        pool_padding = 0
#
#        # create convolutional layers
#        self.hidden_conv_layers = nn.ModuleList()  # list of convolutional layers
#        in_channels = model_params['num_features'][0]
#        n_features = model_params['num_features'][1]
#        for i, out_channels in enumerate(model_params['conv_layer_sizes']):
#            if i > 0:  # apply pooling after the first convolutional layer
#                self.hidden_conv_layers.append(
#                        pool_fn(kernel_size=pool_kernel, stride=pool_stride, padding=pool_padding))
#                n_features = _get_conv1d_length(n_features, pool_kernel,
#                                                stride=pool_stride, padding=pool_padding)
#            self.hidden_conv_layers.append(nn.Sequential(
#                nn.Conv1d(in_channels, out_channels, conv_kernel,
#                          stride=conv_stride, padding=conv_padding),
#                activation_fn,
#                nn.Dropout(p=model_params['dropout']) if model_params['dropout'] else nn.Identity()
#            ).apply(self.init_weights))
#            in_channels = out_channels
#            n_features = _get_conv1d_length(n_features, conv_kernel,
#                                            stride=conv_stride, padding=conv_padding)
#
#        # create linear layers
#        self.flatten = nn.Flatten()  # flattens the input tensor
#        self.hidden_linear_layers = nn.ModuleList()  # list of linear layers
#        in_features = n_features * out_channels
#        for out_features in model_params['dense_layer_sizes']:
#            self.hidden_linear_layers.append(nn.Sequential(
#                nn.Linear(in_features, out_features),
#                activation_fn,
#                nn.Dropout(p=model_params['dropout']) if model_params['dropout'] else nn.Identity()
#            ).apply(self.init_weights))
#            in_features = out_features
#
#        # set output layer
#        self.output_layer = nn.Sequential(
#            nn.Linear(in_features, model_params['num_targets']),
#            nn.Sigmoid()
#        ).apply(self.init_weights)
#
#    def init_weights(self,m):
#        if isinstance(m, nn.Linear):
#            nn.init.xavier_uniform_(m.weight)
#
#    def forward(self, x):
#        for layer in self.hidden_conv_layers:
#            x = layer(x)
#        x = self.flatten(x)
#        for layer in self.hidden_linear_layers:
#            x = layer(x)
#        x = self.output_layer(x)
#        return x
