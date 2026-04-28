"""
Models with transformer layers.

- annotated transformer: <https://nlp.seas.harvard.edu/annotated-transformer/>
- pytorch tutorial: <https://pytorch.org/tutorials/beginner/translation_transformer.html>
"""

import math
import torch
import torch.nn as nn

from .mlp import MLPNet

from typing import Optional, Any, Union, Callable
from torch import Tensor


class DataEmbedding(nn.Module):
    def __init__(self, input_size: int, output_size: int, scale_outputs: bool = False):
        super().__init__()
        self.embedding = nn.Linear(input_size, output_size, bias=False)
        self.output_size = output_size
        self.scale_outputs = scale_outputs
        # initialize parameters
        self.init_parameters()

    def forward(self, x: Tensor):
        r"""
        Maps data from one space to another space (e.g., feature to embedding space).

        Args:
            x: Tensor provided with dimensions (..., feature)
        """
        in_size = x.size()
        x = torch.reshape(x, (-1, in_size[-1]))
        y = self.embedding(x)
        if self.scale_outputs:
            y *= math.sqrt(self.output_size)
        y = torch.reshape(y, (*in_size[:-1], self.output_size))
        return y

    def init_parameters(self):
        r"""Initializes the values of trainable parameters."""
        nn.init.orthogonal_(self.embedding.weight)


class PositionalEmbedding(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        position_max_size: int = 5000,
        use_dropout: Union[bool, float] = False,
    ):
        super().__init__()
        assert position_max_size <= 1.0e4
        # compute the positional encodings once in log space
        den = torch.exp(
            -torch.arange(0, embedding_size, 2) * math.log(1.0e4) / embedding_size
        )
        pos = torch.arange(0, position_max_size).unsqueeze(1)
        pos_embedding = torch.zeros((position_max_size, embedding_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0)
        # set dropout
        if use_dropout:
            self.dropout = nn.Dropout(use_dropout)
        else:
            self.dropout = None
        # register buffer
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, x: Tensor):
        y = x + self.pos_embedding[:, : x.size(1)].requires_grad_(False)
        if self.dropout is not None:
            y = self.dropout(y)
        return y


class Transformer1d0dModel(nn.Module):
    def __init__(
        self,
        src_size: int,
        trg_size: int,
        embedding_size: int,
        n_head: int,
        n_encoder_layers: int = 6,
        n_decoder_layers: int = 6,
        feedforward_size: int = 2048,
        transformer_activation: Callable[[Tensor], Tensor] = nn.ReLU(),
        transformer_kwargs: dict = {},
        output_layer_activation: Optional[Any] = None,
        use_dropout: Union[bool, float] = False,
    ):
        r"""Creates the model.

        Args:
            src_size: Size of features to the encoder (required)
            trg_size: Size of features to the decoder (required)
            embedding_size: Size of embedding space (i.e., dimension of the model)
            n_head: Number of heads in the multiheadattention models
        """
        super().__init__()
        self.trg_size = trg_size
        # create embeddings for input data
        self.encoder_emb = DataEmbedding(src_size, embedding_size)
        self.decoder_emb = DataEmbedding(trg_size, embedding_size)
        # create positional embedding (for encoder data only)
        self.encoder_pos_emb = PositionalEmbedding(
            embedding_size, use_dropout=use_dropout
        )
        # create transformer model
        if use_dropout:
            dropout = use_dropout
        else:
            dropout = 0.0
        self.transformer = nn.Transformer(
            d_model=embedding_size,
            nhead=n_head,
            num_encoder_layers=n_encoder_layers,
            num_decoder_layers=n_decoder_layers,
            dim_feedforward=feedforward_size,
            dropout=dropout,
            activation=transformer_activation,
            batch_first=True,  # tensors provided as (batch, seq, feature)
            **transformer_kwargs,
        )
        # create MLP model for output
        self.generator = MLPNet(
            embedding_size,
            trg_size,
            hidden_layers_sizes=[],
            output_layer_activation=output_layer_activation,
        )

    def forward(self, x_src: Tensor, x_trg: Tensor = None, **kwargs):
        r"""Applies the model function: y = model(x_src, x_trg)

        Args:
            x_src: input tensor
            x_trg: input tensor
        """
        # create zero target if called to generate prediction (i.e., mask target)
        if x_trg is None:
            x_trg = torch.zeros(x_src.size(0), 1, self.trg_size)
        # check input sizes
        assert 3 == x_src.dim()
        assert 3 == x_trg.dim()
        assert 1 == x_trg.size(1)
        # map inputs to space of transformer model
        x_src_emb = self.encoder_pos_emb(self.encoder_emb(x_src))
        x_trg_emb = self.decoder_emb(x_trg)
        # apply transformer
        h = self.transformer(x_src_emb, x_trg_emb, **kwargs)
        # apply output model
        y = self.generator(h)
        return y

    def encode(self, x_src: Tensor, **kwargs):
        x_src_emb = self.encoder_pos_emb(self.encoder_emb(x_src))
        return self.transformer.encoder(x_src_emb, **kwargs)

    def decode(self, x_trg: Tensor, h_src: Tensor, **kwargs):
        x_trg_emb = self.decoder_emb(x_trg)
        return self.transformer.decoder(x_trg_emb, y_src, **kwargs)


###############################################################################

# TODO use doxygen for these test


def test_Transformer1d0dModel():
    emb = DataEmbedding(1, 2)
    pos = PositionalEmbedding(2)
    model = Transformer1d0dModel(
        1,
        3,
        32,
        8,
        n_encoder_layers=2,
        n_decoder_layers=2,
        feedforward_size=128,
        transformer_activation=nn.GELU(),
    )
    print(model)

    print("Test 1:")
    x = torch.tensor([[[1.0], [-1.0], [1.0], [-1.0]]])
    y = emb(x)
    print("- input  x =", x)
    print("- output y =", y)

    print("Test 2:")
    x = torch.tensor([[[1.0, 2.0], [-1.0, -2.0], [1.0, 2.0], [-1.0, -2.0]]])
    y = pos(x)
    print("- input  x =", x)
    print("- output y =", y)

    print("Test 3:")
    x = torch.tensor([[[1.0], [-1.0], [1.0], [-1.0]]])
    y = pos(emb(x))
    print("- input  x =", x)
    print("- output y =", y)

    print("Test 4:")
    x_src = torch.randn(1, 30, 1)
    x_trg = torch.randn(1, 1, 3)
    y = model(x_src, x_trg)
    print("- input  x_src =")
    print(x_src)
    print("- input  x_trg =")
    print(x_trg)
    print("- output y     =")
    print(y)

    print("Test 4:")
    x_src = torch.randn(1, 6, 1)
    y = model(x_src)
    print("- input  x_src =")
    print(x_src)
    print("- output y     =")
    print(y)

    print("----------------------------------------")


if __name__ == "__main__":
    r"""Runs tests."""
    test_Transformer1d0dModel()
