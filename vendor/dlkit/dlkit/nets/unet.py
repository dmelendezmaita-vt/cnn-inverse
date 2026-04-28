"""
UNet
"""

from abc import abstractmethod

import math
import torch
import torch.nn as nn

import dlkit.nets.conv1d
import dlkit.nets.conv2d

# --------------------------------------
# UNet (2025)
# --------------------------------------


def _create_level_blocks(
    input_channels_all,
    output_channels_all,
    kernel_size,
    activation=None,
    dropout=None,
    with_LevelBlock=None,
    with_Normalization=None,
    **conv_kwargs,
):
    block = list()
    for ch_in, ch_out in zip(input_channels_all, output_channels_all):
        block.append(
            with_LevelBlock(
                ch_in,
                ch_out,
                kernel_size,
                **conv_kwargs,
                normalization=with_Normalization(ch_out),
                activation=activation,
                dropout=dropout,
            )
        )
    return block


class UNetXd_2025(nn.Module):
    r"""
    UNet.

    Args:
        input_channels:  number of channels of inputs
        output_channels: number of channels of outputs
    """

    def __init__(
        self,
        input_channels,
        output_channels,
        input_conv_kernels=3,
        input_conv_kwargs={},
        down_levels_conv_channels=[2, 4, 8],
        down_levels_conv_kernels=3,
        down_levels_conv_kwargs={},
        coarse_level_conv_channels=[16, 16],
        coarse_level_conv_kernels=3,
        coarse_level_conv_kwargs={},
        up_levels_conv_channels=[8, 4, 2],
        up_levels_conv_kernels=3,
        up_levels_conv_kwargs={},
        output_conv_kernels=3,
        output_conv_kwargs={},
        hidden_layers_activation=nn.ReLU(),
        output_activation=None,
        use_dropout=False,
        # dimension dependent classes
        with_Downsample=None,
        with_Upsample=None,
        with_LevelBlock=None,
        with_Normalization=None,
    ):
        super().__init__()
        # check dimension dependent classes
        assert with_Downsample is not None
        assert with_Upsample is not None
        assert with_LevelBlock is not None
        assert with_Normalization is not None
        # check channels
        assert isinstance(down_levels_conv_channels, list), type(
            down_levels_conv_channels
        )
        assert isinstance(up_levels_conv_channels, list), type(up_levels_conv_channels)
        assert isinstance(coarse_level_conv_channels, list), type(
            coarse_level_conv_channels
        )
        for l, channels in enumerate(down_levels_conv_channels):
            if not isinstance(channels, list):
                down_levels_conv_channels[l] = [channels]
        for l, channels in enumerate(up_levels_conv_channels):
            if not isinstance(channels, list):
                up_levels_conv_channels[l] = [channels]
        # set number of layers
        assert len(down_levels_conv_channels) == len(up_levels_conv_channels)
        n_levels = len(down_levels_conv_channels)
        # check kernels
        if isinstance(down_levels_conv_kernels, list):
            assert len(down_levels_conv_kernels) == n_levels
        else:  # otherwise assume single integer
            down_levels_conv_kernels = [down_levels_conv_kernels] * n_levels
        if isinstance(up_levels_conv_kernels, list):
            assert len(up_levels_conv_kernels) == n_levels
        else:  # otherwise assume single integer
            up_levels_conv_kernels = [up_levels_conv_kernels] * n_levels
        assert isinstance(coarse_level_conv_kernels, int), type(
            coarse_level_conv_kernels
        )
        # set from arguments
        self.input_channels = input_channels
        self.output_channels = output_channels
        if use_dropout:
            dropout = nn.Dropout(use_dropout)
        else:
            dropout = None
        # TODO manage print statements
        print(f"### {down_levels_conv_channels=}")
        print(f"### {coarse_level_conv_channels=}")
        print(f"### {up_levels_conv_channels=}")
        _indent = ""
        #
        # create input block
        #
        ch_in = input_channels
        ch_out = down_levels_conv_channels[0][0]
        print(f"###{_indent} input {ch_in=}, {ch_out=}")
        self.input_block = with_LevelBlock(
            ch_in,
            ch_out,
            input_conv_kernels,
            **input_conv_kwargs,
            normalization=with_Normalization(ch_out),
            activation=hidden_layers_activation,
            dropout=dropout,
        )
        ch_in = ch_out
        #
        # create downsample levels
        #
        ch_down_all = list()  # initialize channels of all layers (incl. downsample)
        self.down_levels = nn.ModuleList()
        for l, (channels, kernel_size) in enumerate(
            zip(down_levels_conv_channels, down_levels_conv_kernels)
        ):
            _indent = l * "  "
            level = list()
            # add sequence of layers
            ch_in_all = [ch_in] + channels[:-1]  # input channels of all layers
            ch_out_all = channels  # output channels of all layers
            for ci_, co_ in zip(ch_in_all, ch_out_all):
                print(f"###{_indent} down       level={l}, ch_in={ci_}, ch_out={co_}")
            level.extend(
                _create_level_blocks(
                    ch_in_all,
                    ch_out_all,
                    kernel_size,
                    activation=hidden_layers_activation,
                    dropout=dropout,
                    with_LevelBlock=with_LevelBlock,
                    with_Normalization=with_Normalization,
                    **down_levels_conv_kwargs,
                )
            )
            ch_in = ch_out_all[-1]
            # add downsample layer
            # <code id="v1">
            # if l < n_levels - 1:  # if not the last level
            # </code
            # <code id="v2">
            if True:
                # </code
                ch_out = ch_in
                print(f"###{_indent} downsample level={l}, {ch_in=}, {ch_out=}")
                layer = with_Downsample(ch_in, ch_out, kernel_size)
                level.append(layer)
            # add this down level
            self.down_levels.append(nn.ModuleList(level))
        #
        # create coarse level
        #
        _indent = (l + 1) * "  "
        ch_in_all = [ch_in] + coarse_level_conv_channels[
            :-1
        ]  # input channels of all layers
        ch_out_all = coarse_level_conv_channels  # output channels of all layers
        for ci_, co_ in zip(ch_in_all, ch_out_all):
            print(f"###{_indent} coarse     level={l+1}, ch_in={ci_}, ch_out={co_}")
        kernel_size = coarse_level_conv_kernels
        level = _create_level_blocks(
            ch_in_all,
            ch_out_all,
            kernel_size,
            activation=hidden_layers_activation,
            dropout=dropout,
            with_LevelBlock=with_LevelBlock,
            with_Normalization=with_Normalization,
            **coarse_level_conv_kwargs,
        )
        self.coarse_level = nn.Sequential(*level)
        ch_in = ch_out_all[-1]
        #
        # create upsample levels
        #
        self.up_levels = nn.ModuleList()
        for l_inv, (channels, kernel_size) in enumerate(
            zip(up_levels_conv_channels, up_levels_conv_kernels)
        ):
            l = n_levels - 1 - l_inv
            _indent = l * "  "
            level = list()
            # set channels of corresponding down level `l`
            ch_down_inv = list(reversed(down_levels_conv_channels[l]))
            # add upsample layer
            # <code id="v1">
            # if l < n_levels - 1:  # if not the last level
            # </code
            # <code id="v2">
            if True:
                # </code
                ch_in += ch_down_inv[0]
                ch_out = channels[0]
                print(f"###{_indent} upsample   level={l}, {ch_in=} {ch_out=}")
                layer = nn.Sequential(
                    with_LevelBlock(
                        ch_in,
                        ch_out,
                        kernel_size,
                        **up_levels_conv_kwargs,
                        normalization=with_Normalization(ch_out),
                        activation=hidden_layers_activation,
                        dropout=dropout,
                    ),
                    with_Upsample(ch_out, ch_out, kernel_size),
                )
                level.append(layer)
                ch_in = ch_out
            # add sequence of layers
            ch_in_all = [ch_in] + channels[:-1]  # input channels of all layers
            ch_in_all = [
                sum(c) for c in zip(ch_in_all, ch_down_inv)
            ]  # add channels of down level
            ch_out_all = channels  # output channels of all layers
            for ci_, co_ in zip(ch_in_all, ch_out_all):
                print(f"###{_indent} up         level={l}, ch_in={ci_}, ch_out={co_}")
            level.extend(
                _create_level_blocks(
                    ch_in_all,
                    ch_out_all,
                    kernel_size,
                    activation=hidden_layers_activation,
                    dropout=dropout,
                    with_LevelBlock=with_LevelBlock,
                    with_Normalization=with_Normalization,
                    **up_levels_conv_kwargs,
                )
            )
            ch_in = ch_out_all[-1]
            # add this up level
            self.up_levels.append(nn.ModuleList(level))
        #
        # create output block
        #
        ch_out = output_channels
        print(f"###{_indent} output {ch_in=}, {ch_out=}")
        self.output_block = with_LevelBlock(
            ch_in,
            ch_out,
            output_conv_kernels,
            **output_conv_kwargs,
            normalization=with_Normalization(ch_out),
            activation=output_activation,
            dropout=dropout,
        )
        # initialize parameters
        self.init_parameters()

    def forward(self, x):
        r"""Applies the network's forward function: y = net(x)

        Args:
            x: input tensor
        """
        assert x.size(1) == self.input_channels
        # input layer
        h = self.input_block(x)
        # downsample levels
        h_down = list()
        for level in self.down_levels:
            for block in level:
                h = block(h)
                h_down.append(h)
        # coarse level
        h = self.coarse_level(h)
        # upsample levels
        for level in self.up_levels:
            for block in level:
                h_cat = torch.cat(
                    [h, h_down.pop()], dim=1
                )  # concatenate along channel dimension
                h = block(h_cat)
        # output layer
        y = self.output_block(h)
        return y

    def init_parameters(self):
        r"""Initializes the values of trainable parameters."""
        # input layer
        self.input_block.init_parameters()
        # downsample levels
        for level in self.down_levels:
            for block in level:
                try:
                    block.init_parameters()
                except AttributeError:
                    for layer in block:
                        layer.init_parameters()
        # coarse level
        try:
            self.coarse_level.init_parameters()
        except AttributeError:
            for block in self.coarse_level:
                block.init_parameters()
        # upsample levels
        for level in self.up_levels:
            for block in level:
                try:
                    block.init_parameters()
                except AttributeError:
                    for layer in block:
                        layer.init_parameters()
        # output layer
        self.output_block.init_parameters()


# --------------------------------------


class UNet2d_2025(UNetXd_2025):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            with_Downsample=dlkit.nets.conv2d.Downsample,
            with_Upsample=dlkit.nets.conv2d.Upsample,
            with_LevelBlock=dlkit.nets.conv2d.LevelBlock,
            with_Normalization=dlkit.nets.conv2d.Normalization,
        )


# --------------------------------------
# UNet (2021) <https://github.com/openai/improved-diffusion>
# --------------------------------------


def _zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    Compute embedding `emb` as:

        fq[j]  = exp( -log(T_max) * j/(d/2) ),  for j=0,..,d/2
        x[i,j] = ts[i] * fq[j],  for i=0,..,N, j=0,..,d/2
        emb = [cos(x) , sin(x)]

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class EmbedModule(nn.Module):
    """
    Any module where forward() takes embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` embeddings.
        """


class EmbedSequential(EmbedModule, nn.Sequential):
    """
    A sequential module that passes embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, EmbedModule):
                assert emb is not None
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class ResBlock2d_EmbedBlock(dlkit.nets.conv2d.ResBlock, EmbedModule):
    """
    A residual block that can optionally change the number of channels.

    :param input_channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param output_channels: if specified, the number of out channels.
    :param use_conv: if True and output_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    """

    def __init__(
        self,
        input_channels,
        emb_channels,
        output_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        normalization=None,
    ):
        super().__init__(
            input_channels,
            output_channels=output_channels,
            use_conv=use_conv,
            normalization=normalization,
        )
        self.emb_channels = emb_channels
        self.use_scale_shift_norm = use_scale_shift_norm
        # create embedding layers
        if use_scale_shift_norm:
            emb_out_channels = 2 * self.output_channels
        else:
            emb_out_channels = self.output_channels
        if emb_channels is not None:
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                nn.Linear(emb_channels, emb_out_channels),
            )

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert x.size(0) == emb.size(0)
        assert x.size(1) == self.input_channels
        assert emb.size(1) == self.emb_channels
        # apply input layers
        h = self.in_layers(x)
        # apply embedding layers
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        # apply output layers
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            h = out_norm(h)
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = h * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        # apply skip connection
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Computes:

        q,k,v = conv1d(norm(x))
        h = qkv_attention(q, k, v)
        h = conv1d(h)
        y = x + h
    """

    def __init__(self, channels, num_heads=1, normalization=None):
        super().__init__()
        self.channels = channels
        self.num_heads = (
            num_heads  # TODO probably num_heads need to be used in dim of self.qkv
        )

        self.norm = normalization(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.attention = QKVAttention()
        self.proj_out = _zero_module(nn.Conv1d(channels, channels, 1))

    def forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(b * self.num_heads, -1, qkv.shape[2])
        h = self.attention(qkv)
        h = h.reshape(b, -1, h.shape[-1])
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention.
    """

    def forward(self, qkv):
        """
        Apply QKV attention.

        Input: q, k, v
        Computes:

            s = (#channels)^(-1/4)
            y = softmax(q*s * k^T*s) * v

        :param qkv: an [N x (C * 3) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x C x T] tensor after attention.
        """
        ch = qkv.shape[1] // 3
        q, k, v = torch.split(qkv, ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        return torch.einsum("bts,bcs->bct", weight, v)

    @staticmethod
    def count_flops(model, _x, y):
        """
        A counter for the `thop` package to count the operations in an
        attention operation.

        Meant to be used like:

            macs, params = thop.profile(
                model,
                inputs=(inputs, timestamps),
                custom_ops={QKVAttention: QKVAttention.count_flops},
            )

        """
        b, c, *spatial = y[0].shape
        num_spatial = int(np.prod(spatial))
        # We perform two matmuls with the same number of ops.
        # The first computes the weight matrix, the second computes
        # the combination of the value vectors.
        matmul_ops = 2 * b * (num_spatial**2) * c
        model.total_ops += torch.DoubleTensor([matmul_ops])


class UNetXd_2021_idd(nn.Module):
    r"""
    UNet.

    Args:
        input_channels:  number of channels of inputs
        output_channels: number of channels of outputs
        internal_channels: base channel count for the model.
        num_res_blocks: number of residual blocks per downsample.
        attention_levels: a collection of levels at which
            attention will take place. May be a set, list, or tuple.
            For example, if this contains 4, then at 4x downsampling, attention
            will be used.

        channel_mult: channel multiplier for each level of the UNet.
        num_classes: if specified (as an int), then this model will be
            class-conditional with `num_classes` classes.
        num_heads: the number of attention heads in each attention layer.
    """

    def __init__(
        self,
        input_channels,
        output_channels,
        internal_channels,
        num_res_blocks=1,
        attention_levels=[],
        channel_mult=(1, 2, 4, 8),
        num_classes=None,
        num_heads=1,
        num_heads_upsample=-1,
        time_embed_dim=None,
        # dimension dependent classes
        with_InputLayer=None,
        with_OutputLayer=None,
        with_Downsample=None,
        with_Upsample=None,
        with_LevelBlock=None,
        with_Normalization=None,
    ):
        super().__init__()
        # check dimension dependent classes
        assert with_InputLayer is not None
        assert with_OutputLayer is not None
        assert with_Downsample is not None
        assert with_Upsample is not None
        assert with_LevelBlock is not None
        assert with_Normalization is not None
        # set from arguments
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.internal_channels = internal_channels
        self.num_res_blocks = num_res_blocks
        self.attention_levels = attention_levels
        self.channel_mult = channel_mult
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.time_embed_dim = time_embed_dim
        if -1 == num_heads_upsample:
            self.num_heads_upsample = num_heads
        else:
            self.num_heads_upsample = num_heads_upsample
        # create embedding layers
        if time_embed_dim is not None:
            # create time embedding layers
            self.time_embed = nn.Sequential(
                nn.Linear(internal_channels, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim),
            )
            # create label embedding layers
            if self.num_classes is not None:
                self.label_emb = nn.Embedding(num_classes, time_embed_dim)
        #
        # create downsample blocks
        #
        input_layer = EmbedSequential(
            with_InputLayer(input_channels, internal_channels)
        )
        self.input_blocks = nn.ModuleList([input_layer])
        input_block_channels = [internal_channels]
        ch = internal_channels
        for level, mult in enumerate(channel_mult):
            print(f"### downsample {level=}")
            for _ in range(num_res_blocks):
                layers = [
                    with_LevelBlock(
                        ch,
                        output_channels=mult * internal_channels,
                        normalization=with_Normalization,
                    )
                ]
                ch = mult * internal_channels
                if level in attention_levels:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=self.num_heads,
                            normalization=with_Normalization,
                        )
                    )
                self.input_blocks.append(EmbedSequential(*layers))
                input_block_channels.append(ch)
            if level != len(channel_mult) - 1:
                down_layer = EmbedSequential(with_Downsample(ch, ch, 3))
                self.input_blocks.append(down_layer)
                input_block_channels.append(ch)
        #
        # create middle block
        #
        level += 1
        print(f"### middle {level=}")
        layers = [with_LevelBlock(ch, normalization=with_Normalization)]
        if level in attention_levels:
            layers.append(
                AttentionBlock(
                    ch, num_heads=self.num_heads, normalization=with_Normalization
                )
            )
        layers.append(with_LevelBlock(ch, normalization=with_Normalization))
        self.middle_block = EmbedSequential(*layers)
        #
        # create upsample blocks
        #
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            print(f"### upsample {level=}")
            for i in range(num_res_blocks + 1):
                layers = [
                    with_LevelBlock(
                        ch + input_block_channels.pop(),
                        output_channels=mult * internal_channels,
                        normalization=with_Normalization,
                    )
                ]
                ch = mult * internal_channels
                if level in attention_levels:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=self.num_heads_upsample,
                            normalization=with_Normalization,
                        )
                    )
                if level and i == num_res_blocks:
                    layers.append(with_Upsample(ch, ch, 3))
                self.output_blocks.append(EmbedSequential(*layers))
        #
        # create output layer
        #
        self.output_layer = nn.Sequential(
            with_Normalization(ch),
            nn.SiLU(),
            with_OutputLayer(ch, output_channels),
        )

    def forward(self, x, timesteps=None, y=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if timesteps is not None:
            assert (
                self.time_embed_dim is not None
            ), "must specify timestep if and only if the model has time embedding"
            assert x.size(0) == timesteps.size(0)
            assert timesteps.dim() == 1
            if y is not None:
                assert (
                    self.num_classes is not None
                ), "must specify y if and only if the model is class-conditional"
                assert y.shape == (x.shape[0],)
        # apply embedding layers
        if timesteps is not None:
            emb = self.time_embed(timestep_embedding(timesteps, self.internal_channels))
            if y is not None:
                emb = emb + self.label_emb(y)
        else:
            emb = None
        # apply layers
        hs = []
        h = x.type(self.inner_dtype)
        for block in self.input_blocks:
            h = block(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for block in self.output_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = block(cat_in, emb)
        h = h.type(x.dtype)
        return self.output_layer(h)

    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return next(self.input_blocks.parameters()).dtype


class UNet1d_2021(UNetXd_2021_idd):
    def __init__(self, *args, internal_channels=32, **kwargs):
        def _with_InputLayer(_input_channels, _internal_channels):
            return nn.Conv1d(
                _input_channels,
                _internal_channels,
                3,
                padding=1,
                padding_mode="replicate",
            )

        def _with_OutputLayer(_internal_channels, _output_channels):
            return _zero_module(
                nn.Conv1d(
                    _internal_channels,
                    _output_channels,
                    3,
                    padding=1,
                    padding_mode="replicate",
                )
            )

        def _Normalization(_num_channels):
            return dlkit.nets.conv1d.Normalization(
                _num_channels, num_groups=internal_channels
            )

        super().__init__(
            *args,
            internal_channels=internal_channels,
            with_InputLayer=_with_InputLayer,
            with_OutputLayer=_with_OutputLayer,
            with_Downsample=dlkit.nets.conv1d.UNetDownsample,
            with_Upsample=dlkit.nets.conv1d.UNetUpsample,
            with_LevelBlock=dlkit.nets.conv1d.UNetResBlock,
            with_Normalization=_Normalization,
            **kwargs,
        )


class UNet2d_2021(UNetXd_2021_idd):
    def __init__(self, *args, internal_channels=32, **kwargs):
        def _with_InputLayer(_input_channels, _internal_channels):
            return nn.Conv2d(
                _input_channels,
                _internal_channels,
                3,
                padding=1,
                padding_mode="replicate",
            )

        def _with_OutputLayer(_internal_channels, _output_channels):
            return _zero_module(
                nn.Conv2d(
                    _internal_channels,
                    _output_channels,
                    3,
                    padding=1,
                    padding_mode="replicate",
                )
            )

        def _Normalization(_num_channels):
            return dlkit.nets.conv2d.Normalization(
                _num_channels, num_groups=internal_channels
            )

        super().__init__(
            *args,
            internal_channels=internal_channels,
            with_InputLayer=_with_InputLayer,
            with_OutputLayer=_with_OutputLayer,
            with_Downsample=dlkit.nets.conv2d.Downsample,
            with_Upsample=dlkit.nets.conv2d.Upsample,
            with_LevelBlock=dlkit.nets.conv2d.ResBlock,
            with_Normalization=_Normalization,
            **kwargs,
        )


class UNet2d_2021_idd(UNetXd_2021_idd):
    def __init__(self, *args, internal_channels=32, **kwargs):
        time_embed_dim = 4 * internal_channels

        def _with_InputLayer(_input_channels, _internal_channels):
            return nn.Conv2d(
                _input_channels,
                _internal_channels,
                3,
                padding=1,
                padding_mode="replicate",
            )

        def _with_OutputLayer(_internal_channels, _output_channels):
            return _zero_module(
                nn.Conv2d(
                    _internal_channels,
                    _output_channels,
                    3,
                    padding=1,
                    padding_mode="replicate",
                )
            )

        def _ResBlock2d_EmbedBlock(_input_channels, **_kwargs):
            return ResBlock2d_EmbedBlock(_input_channels, time_embed_dim, **_kwargs)

        def _Normalization(_num_channels):
            return dlkit.nets.conv2d.Normalization(_num_channels, num_groups=32)

        super().__init__(
            *args,
            internal_channels=internal_channels,
            time_embed_dim=time_embed_dim,
            with_InputLayer=_with_InputLayer,
            with_OutputLayer=_with_OutputLayer,
            with_Downsample=dlkit.nets.conv2d.Downsample,
            with_Upsample=dlkit.nets.conv2d.Upsample,
            with_LevelBlock=_ResBlock2d_EmbedBlock,
            with_Normalization=_Normalization,
            **kwargs,
        )


# --------------------------------------
# Encoder & Decoder Nets (2021)
# --------------------------------------


class EncoderNetXd_2021_idd(nn.Module):
    r"""
    Encoder Net derived from UNetXd_2021_idd.

    Args:
        input_channels:  number of channels of inputs
        output_channels: number of channels of outputs
        internal_channels: base channel count for the model.
        num_res_blocks: number of residual blocks per downsample.
        attention_levels: a collection of levels at which
            attention will take place. May be a set, list, or tuple.
            For example, if this contains 4, then at 4x downsampling, attention
            will be used.

        channel_mult: channel multiplier for each level of the UNet.
        num_classes: if specified (as an int), then this model will be
            class-conditional with `num_classes` classes.
        num_heads: the number of attention heads in each attention layer.
    """

    def __init__(
        self,
        input_channels,
        output_channels,
        internal_channels,
        num_res_blocks=1,
        attention_levels=[],
        channel_mult=(1, 1, 1, 1),
        num_classes=None,
        num_heads=1,
        num_heads_upsample=-1,
        time_embed_dim=None,
        # dimension dependent classes
        with_InputLayer=None,
        with_OutputLayer=None,
        with_Downsample=None,
        with_Upsample=None,
        with_LevelBlock=None,
        with_Normalization=None,
    ):
        super().__init__()
        # check dimension dependent classes
        assert with_InputLayer is not None
        assert with_OutputLayer is not None
        assert with_Downsample is not None
        assert with_Upsample is not None
        assert with_LevelBlock is not None
        assert with_Normalization is not None
        # set from arguments
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.internal_channels = internal_channels
        self.num_res_blocks = num_res_blocks
        self.attention_levels = attention_levels
        self.channel_mult = channel_mult
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.time_embed_dim = time_embed_dim
        if -1 == num_heads_upsample:
            self.num_heads_upsample = num_heads
        else:
            self.num_heads_upsample = num_heads_upsample
        # create embedding layers
        if time_embed_dim is not None:
            # create time embedding layers
            self.time_embed = nn.Sequential(
                nn.Linear(internal_channels, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim),
            )
            # create label embedding layers
            if self.num_classes is not None:
                self.label_emb = nn.Embedding(num_classes, time_embed_dim)
        #
        # create downsample blocks
        #
        input_layer = EmbedSequential(
            with_InputLayer(input_channels, internal_channels)
        )
        self.input_blocks = nn.ModuleList([input_layer])
        ch = internal_channels
        for level, mult in enumerate(channel_mult):
            print(f"### downsample {level=}")
            for _ in range(num_res_blocks):
                layers = [
                    with_LevelBlock(
                        ch,
                        output_channels=mult * internal_channels,
                        normalization=with_Normalization,
                    )
                ]
                ch = mult * internal_channels
                if level in attention_levels:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=self.num_heads,
                            normalization=with_Normalization,
                        )
                    )
                self.input_blocks.append(EmbedSequential(*layers))
            if level != len(channel_mult) - 1:
                down_layer = EmbedSequential(with_Downsample(ch, ch, 3))
                self.input_blocks.append(down_layer)
        #
        # create middle block
        #
        level = len(channel_mult)
        print(f"### middle {level=}")
        layers = [with_LevelBlock(ch, normalization=with_Normalization)]
        self.middle_block = EmbedSequential(*layers)
        #
        # create output layer
        #
        self.output_layer = nn.Sequential(
            with_Normalization(ch),
            nn.SiLU(),
            with_OutputLayer(ch, output_channels),
        )

    def forward(self, x, timesteps=None, y=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if timesteps is not None:
            assert (
                self.time_embed_dim is not None
            ), "must specify timestep if and only if the model has time embedding"
            assert x.size(0) == timesteps.size(0)
            assert timesteps.dim() == 1
            if y is not None:
                assert (
                    self.num_classes is not None
                ), "must specify y if and only if the model is class-conditional"
                assert y.shape == (x.shape[0],)
        # apply embedding layers
        if timesteps is not None:
            emb = self.time_embed(timestep_embedding(timesteps, self.internal_channels))
            if y is not None:
                emb = emb + self.label_emb(y)
        else:
            emb = None
        # apply layers
        h = x.type(self.inner_dtype)
        for block in self.input_blocks:
            h = block(h, emb)
        h = self.middle_block(h, emb)
        return self.output_layer(h)

    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return next(self.input_blocks.parameters()).dtype


class DecoderNetXd_2021_idd(nn.Module):
    r"""
    Decoder Net derived from UNetXd_2021_idd.

    Args:
        input_channels:  number of channels of inputs
        output_channels: number of channels of outputs
        internal_channels: base channel count for the model.
        num_res_blocks: number of residual blocks per downsample.
        attention_levels: a collection of levels at which
            attention will take place. May be a set, list, or tuple.
            For example, if this contains 4, then at 4x downsampling, attention
            will be used.

        channel_mult: channel multiplier for each level of the UNet.
        num_classes: if specified (as an int), then this model will be
            class-conditional with `num_classes` classes.
        num_heads: the number of attention heads in each attention layer.
    """

    def __init__(
        self,
        input_channels,
        output_channels,
        internal_channels,
        num_res_blocks=1,
        attention_levels=[],
        channel_mult=(1, 1, 1, 1),
        num_classes=None,
        num_heads=1,
        num_heads_upsample=-1,
        time_embed_dim=None,
        # dimension dependent classes
        with_InputLayer=None,
        with_OutputLayer=None,
        with_Downsample=None,
        with_Upsample=None,
        with_LevelBlock=None,
        with_Normalization=None,
    ):
        super().__init__()
        # check dimension dependent classes
        assert with_InputLayer is not None
        assert with_OutputLayer is not None
        assert with_Downsample is not None
        assert with_Upsample is not None
        assert with_LevelBlock is not None
        assert with_Normalization is not None
        # set from arguments
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.internal_channels = internal_channels
        self.num_res_blocks = num_res_blocks
        self.attention_levels = attention_levels
        self.channel_mult = channel_mult
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.time_embed_dim = time_embed_dim
        if -1 == num_heads_upsample:
            self.num_heads_upsample = num_heads
        else:
            self.num_heads_upsample = num_heads_upsample
        # create embedding layers
        if time_embed_dim is not None:
            # create time embedding layers
            self.time_embed = nn.Sequential(
                nn.Linear(internal_channels, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim),
            )
            # create label embedding layers
            if self.num_classes is not None:
                self.label_emb = nn.Embedding(num_classes, time_embed_dim)
        #
        # create middle block
        #
        ch = channel_mult[-1] * internal_channels
        self.input_layer = EmbedSequential(with_InputLayer(input_channels, ch))
        level = len(channel_mult)
        print(f"### middle {level=}")
        layers = [with_LevelBlock(ch, normalization=with_Normalization)]
        self.middle_block = EmbedSequential(*layers)
        #
        # create upsample blocks
        #
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            print(f"### upsample {level=}")
            for i in range(num_res_blocks + 1):
                layers = [
                    with_LevelBlock(
                        ch,
                        output_channels=mult * internal_channels,
                        normalization=with_Normalization,
                    )
                ]
                ch = mult * internal_channels
                if level in attention_levels:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=self.num_heads_upsample,
                            normalization=with_Normalization,
                        )
                    )
                if level and i == num_res_blocks:
                    layers.append(with_Upsample(ch, ch, 3))
                self.output_blocks.append(EmbedSequential(*layers))
        #
        # create output layer
        #
        self.output_layer = nn.Sequential(
            with_Normalization(ch),
            nn.SiLU(),
            with_OutputLayer(ch, output_channels),
        )

    def forward(self, x, timesteps=None, y=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if timesteps is not None:
            assert (
                self.time_embed_dim is not None
            ), "must specify timestep if and only if the model has time embedding"
            assert x.size(0) == timesteps.size(0)
            assert timesteps.dim() == 1
            if y is not None:
                assert (
                    self.num_classes is not None
                ), "must specify y if and only if the model is class-conditional"
                assert y.shape == (x.shape[0],)
        # apply embedding layers
        if timesteps is not None:
            emb = self.time_embed(timestep_embedding(timesteps, self.internal_channels))
            if y is not None:
                emb = emb + self.label_emb(y)
        else:
            emb = None
        # apply layers
        h = x.type(self.inner_dtype)
        h = self.input_layer(h, emb)
        h = self.middle_block(h, emb)
        for block in self.output_blocks:
            h = block(h, emb)
        h = h.type(x.dtype)
        return self.output_layer(h)

    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return next(self.output_blocks.parameters()).dtype


class EncoderNet1d_2021(EncoderNetXd_2021_idd):
    def __init__(self, *args, internal_channels=32, **kwargs):
        def _with_InputLayer(_input_channels, _internal_channels):
            return nn.Conv1d(
                _input_channels,
                _internal_channels,
                3,
                padding=1,
                padding_mode="replicate",
            )

        def _with_OutputLayer(_internal_channels, _output_channels):
            return _zero_module(
                nn.Conv1d(
                    _internal_channels,
                    _output_channels,
                    3,
                    padding=1,
                    padding_mode="replicate",
                )
            )

        def _Normalization(_num_channels):
            return dlkit.nets.conv1d.Normalization(
                _num_channels, num_groups=internal_channels
            )

        super().__init__(
            *args,
            internal_channels=internal_channels,
            with_InputLayer=_with_InputLayer,
            with_OutputLayer=_with_OutputLayer,
            with_Downsample=dlkit.nets.conv1d.UNetDownsample,
            with_Upsample=dlkit.nets.conv1d.UNetUpsample,
            with_LevelBlock=dlkit.nets.conv1d.UNetResBlock,
            with_Normalization=_Normalization,
            **kwargs,
        )


class DecoderNet1d_2021(DecoderNetXd_2021_idd):
    def __init__(self, *args, internal_channels=32, **kwargs):
        def _with_InputLayer(_input_channels, _internal_channels):
            return nn.Conv1d(
                _input_channels,
                _internal_channels,
                3,
                padding=1,
                padding_mode="replicate",
            )

        def _with_OutputLayer(_internal_channels, _output_channels):
            return _zero_module(
                nn.Conv1d(
                    _internal_channels,
                    _output_channels,
                    3,
                    padding=1,
                    padding_mode="replicate",
                )
            )

        def _Normalization(_num_channels):
            return dlkit.nets.conv1d.Normalization(
                _num_channels, num_groups=internal_channels
            )

        super().__init__(
            *args,
            internal_channels=internal_channels,
            with_InputLayer=_with_InputLayer,
            with_OutputLayer=_with_OutputLayer,
            with_Downsample=dlkit.nets.conv1d.UNetDownsample,
            with_Upsample=dlkit.nets.conv1d.UNetUpsample,
            with_LevelBlock=dlkit.nets.conv1d.UNetResBlock,
            with_Normalization=_Normalization,
            **kwargs,
        )


# --------------------------------------
# Tests
# --------------------------------------

# TODO use doxygen for these test


def test_UNet2d_2025():
    print("---------------------------------------^")
    net = UNet2d_2025(
        1,
        1,
        down_levels_conv_channels=[[2, 2], [4, 4], [8, 8]],
        up_levels_conv_channels=[[8, 8], [4, 4], [2, 2]],
    )
    print(net)

    print("Test 1:")
    x = torch.ones((1, 1, 16, 16))
    y = net(x)
    print("- input  x =", x, sep="\n")
    print("- output y =", y, sep="\n")
    print("---------------------------------------$")


def test_UNet1d_2021():
    print("---------------------------------------^")
    net = UNet1d_2021(1, 1, channel_mult=(1, 2, 4))
    print(net)

    print("Test 1:")
    x = torch.ones((1, 1, 16))
    y = net(x)
    print("- input  x =", x, sep="\n")
    print("- output y =", y, sep="\n")
    print("---------------------------------------$")


def test_UNet2d_2021():
    print("---------------------------------------^")
    net = UNet2d_2021(1, 1, channel_mult=(1, 2, 4))
    print(net)

    print("Test 1:")
    x = torch.ones((1, 1, 16, 16))
    y = net(x)
    print("- input  x =", x, sep="\n")
    print("- output y =", y, sep="\n")
    print("---------------------------------------$")


def test_UNet2d_2021_idd():
    print("---------------------------------------^")
    net = UNet2d_2021_idd(1, 1, channel_mult=(1, 2, 4))
    print(net)

    print("Test 1:")
    x = torch.ones((1, 1, 16, 16))
    t = torch.tensor([0.1])
    y = net(x, t)
    print("- input  x =", x, sep="\n")
    print("- input  t =", t, sep="\n")
    print("- output y =", y, sep="\n")
    print("---------------------------------------$")


def test_EncoderNet1d_2021():
    print("---------------------------------------^")
    net = EncoderNet1d_2021(1, 1, channel_mult=(1, 1, 1))
    print(net)

    print("Test 1:")
    x = torch.ones((1, 1, 16))
    y = net(x)
    print("- input  x =", x, sep="\n")
    print("- output y =", y, sep="\n")
    print("---------------------------------------$")


def test_DecoderNet1d_2021():
    print("---------------------------------------^")
    net = DecoderNet1d_2021(1, 1, channel_mult=(1, 1, 1))
    print(net)

    print("Test 1:")
    x = torch.ones((1, 1, 4))
    y = net(x)
    print("- input  x =", x, sep="\n")
    print("- output y =", y, sep="\n")
    print("---------------------------------------$")


if __name__ == "__main__":
    r"""Runs tests."""
    test_UNet2d_2025()
    test_UNet1d_2021()
    test_UNet2d_2021()
    test_UNet2d_2021_idd()
    test_EncoderNet1d_2021()
    test_DecoderNet1d_2021()
