"""
Networks with 1D transformer layers.
"""

import math
import torch
import torch.nn as nn

# --------------------------------------
# Embeddings
# --------------------------------------


class PatchEmbedding1D(nn.Module):
    """Convert 1D time series into patches and embed them"""

    def __init__(self, patch_size, embedding_size):
        super().__init__()
        self.patch_size = patch_size

        # Linear projection to embed each patch
        self.projection = nn.Linear(patch_size, embedding_size)

    def forward(self, x):
        # x size: (batch_size, input_seq_size)
        batch_size = x.size(0)

        # Reshape into patches: (batch_size, num_patches, patch_size)
        x = x.view(batch_size, -1, self.patch_size)

        # Project each patch to embedding dimension
        x = self.projection(x)  # -> (batch_size, num_patches, embedding_size)

        return x


class ChannelWisePatchEmbedding1D(nn.Module):
    """Embedding that processes each channel separately then combines"""

    def __init__(self, patch_size, embedding_size, n_channels=1, combine_channels=True):
        super().__init__()
        self.patch_size = patch_size
        self.n_channels = n_channels
        self.combine_channels = combine_channels

        assert (
            embedding_size % n_channels == 0
        ), f"embedding size is not divisible by the number of channels`: {embedding_size=}, {n_channels=}"

        # Separate projection for each channel
        self.channel_projections = nn.ModuleList(
            [
                nn.Linear(patch_size, embedding_size // n_channels)
                for _ in range(n_channels)
            ]
        )

        # Final projection to combine channels
        if self.combine_channels:
            self.combine_projection = nn.Linear(embedding_size, embedding_size)

    def forward(self, x):
        # x size: (batch_size, n_channels, input_seq_size)
        batch_size, n_channels, seq_size = x.size()

        channel_embeddings = []

        for i in range(n_channels):
            # Extract channel i: (batch_size, input_seq_size)
            channel_data = x[:, i, :]

            # Reshape into patches: (batch_size, num_patches, patch_size)
            patches = channel_data.view(batch_size, -1, self.patch_size)

            # Project patches for this channel
            embedded = self.channel_projections[i](patches)
            channel_embeddings.append(embedded)

        # Concatenate channel embeddings
        x = torch.cat(
            channel_embeddings, dim=-1
        )  # (batch_size, num_patches, embedding_size)

        # Final projection
        if self.combine_channels:
            x = self.combine_projection(x)

        return x


# --------------------------------------
# Attention Blocks
# --------------------------------------


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_size, n_heads, dropout=0.1):
        super().__init__()
        self.embedding_size = embedding_size
        self.head_size = embedding_size // n_heads
        self.n_heads = n_heads

        assert (
            embedding_size % n_heads == 0
        ), f"embedding size is not divisible by the number of heads: {embedding_size=}, {n_heads=}"

        self.qkv = nn.Linear(embedding_size, embedding_size * 3)
        self.projection = nn.Linear(embedding_size, embedding_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, embedding_size = x.shape

        # Generate Q, K, V
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.n_heads, self.head_size)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, n_heads, seq_len, head_size)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_size)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size, seq_len, embedding_size
        )

        return self.projection(attn_output)


class ChannelWiseMultiHeadAttention(nn.Module):
    """Multi-head attention that processes each channel separately then combines"""

    def __init__(self, embedding_size, n_heads_per_channel, n_channels, dropout=0.1):
        super().__init__()
        self.embedding_size = embedding_size
        self.channel_embed_size = embedding_size // n_channels
        self.n_heads_per_channel = n_heads_per_channel
        self.n_channels = n_channels

        assert (
            embedding_size % n_channels == 0
        ), f"embedding size is not divisible by the number of channels: {embedding_size=}, {n_channels=}"
        assert (
            self.channel_embed_size % n_heads_per_channel == 0
        ), f"embedding size is not divisible by the number of heads: {self.channel_embed_size=}, {n_heads_per_channel=}"

        # Separate QKV projections for each channel
        self.channel_qkv = nn.ModuleList(
            [
                nn.Linear(self.channel_embed_size, self.channel_embed_size * 3)
                for _ in range(n_channels)
            ]
        )

        # Separate projections for each channel
        self.channel_projections = nn.ModuleList(
            [
                nn.Linear(self.channel_embed_size, self.channel_embed_size)
                for _ in range(n_channels)
            ]
        )

        # Final combination projection
        self.combine_projection = nn.Linear(embedding_size, embedding_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, embedding_size = x.shape

        # Split input into channels
        # Assume the embedding is organized as [ch1_features, ch2_features, ...]
        channel_outputs = []

        for ch in range(self.n_channels):
            # Extract channel features
            start_idx = ch * self.channel_embed_size
            end_idx = (ch + 1) * self.channel_embed_size
            x_ch = x[
                :, :, start_idx:end_idx
            ]  # (batch_size, seq_len, channel_embed_size)

            # Generate Q, K, V for this channel
            qkv_ch = self.channel_qkv[ch](
                x_ch
            )  # (batch_size, seq_len, channel_embed_size * 3)
            qkv_ch = qkv_ch.reshape(
                batch_size,
                seq_len,
                3,
                self.n_heads_per_channel,
                self.channel_embed_size // self.n_heads_per_channel,
            )
            qkv_ch = qkv_ch.permute(
                2, 0, 3, 1, 4
            )  # (3, batch_size, n_heads_per_channel, seq_len, head_size_per_channel)
            q_ch, k_ch, v_ch = qkv_ch[0], qkv_ch[1], qkv_ch[2]

            # Attention for this channel
            scores_ch = torch.matmul(q_ch, k_ch.transpose(-2, -1)) / math.sqrt(
                self.channel_embed_size // self.n_heads_per_channel
            )
            attn_weights_ch = torch.softmax(scores_ch, dim=-1)
            attn_weights_ch = self.dropout(attn_weights_ch)

            # Apply attention to values
            attn_output_ch = torch.matmul(attn_weights_ch, v_ch)

            # Concatenate heads for this channel
            attn_output_ch = attn_output_ch.transpose(1, 2).reshape(
                batch_size, seq_len, self.channel_embed_size
            )

            # Channel-specific projection
            attn_output_ch = self.channel_projections[ch](attn_output_ch)
            channel_outputs.append(attn_output_ch)

        # Concatenate all channel outputs
        combined_output = torch.cat(
            channel_outputs, dim=-1
        )  # (batch_size, seq_len, embedding_size)

        # Final combination projection
        output = self.combine_projection(combined_output)

        return output


# --------------------------------------
# Transformer Nets
# --------------------------------------


class TransformerBlock(nn.Module):
    def __init__(self, embedding_size, attn_n_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.attention_block = nn.Sequential(
            nn.LayerNorm(embedding_size),
            MultiHeadAttention(embedding_size, attn_n_heads, dropout),
        )

        activation_size = int(embedding_size * mlp_ratio)
        self.activation_block = nn.Sequential(
            nn.LayerNorm(embedding_size),
            nn.Linear(embedding_size, activation_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(activation_size, embedding_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # apply attention block with residual connection
        h = x + self.attention_block(x)

        # apply activation block with residual connection
        y = h + self.activation_block(h)
        return y


class ChannelWiseTransformerBlock(nn.Module):
    """Transformer block with channel-wise attention"""

    def __init__(
        self, embedding_size, attn_n_heads, n_channels, mlp_ratio=4, dropout=0.1
    ):
        super().__init__()
        self.attention_block = nn.Sequential(
            nn.LayerNorm(embedding_size),
            ChannelWiseMultiHeadAttention(
                embedding_size, attn_n_heads, n_channels, dropout
            ),
        )

        activation_size = int(embedding_size * mlp_ratio)
        self.activation_block = nn.Sequential(
            nn.LayerNorm(embedding_size),
            nn.Linear(embedding_size, activation_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(activation_size, embedding_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # apply attention block with residual connection
        h = x + self.attention_block(x)

        # apply activation block with residual connection
        y = h + self.activation_block(h)
        return y


class TransformerNet(nn.Module):
    """Transformer network derived from ViT architecture."""

    def __init__(
        self,
        input_seq_size,  # Length of input time series
        output_size,  # Output dimension
        patch_size=50,  # Size of each patch
        embedding_size=256,  # Embedding dimension
        attn_n_heads=6 * [8],  # Number of attention heads for each transformer block
        dropout=0.1,
    ):
        super().__init__()

        # Patch embedding
        self.patch_embed = PatchEmbedding1D(patch_size, embedding_size)
        num_patches = input_seq_size // patch_size

        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches + 1, embedding_size) * 0.02
        )

        # Class token (like in original ViT)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_size) * 0.02)

        # Transformer blocks
        self.transformer = nn.ModuleList(
            [
                TransformerBlock(embedding_size, nh, dropout=dropout)
                for nh in attn_n_heads
            ]
        )

        # Final layer norm
        self.norm = nn.LayerNorm(embedding_size)

        # Classification head - outputs 2D vector
        self.head = nn.Linear(embedding_size, output_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch_size, input_seq_size)
        batch_size = x.shape[0]

        # Create patches and embed
        x = self.patch_embed(x)  # (batch_size, num_patches, embedding_size)

        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat(
            [cls_tokens, x], dim=1
        )  # (batch_size, num_patches + 1, embedding_size)

        # Add positional embedding
        x = x + self.pos_embed
        x = self.dropout(x)

        # Apply transformer blocks
        for transformer_block in self.transformer:
            x = transformer_block(x)

        # Layer norm
        x = self.norm(x)

        # Use class token for final prediction
        cls_token_final = x[:, 0]  # (batch_size, embedding_size)

        # Output 2D vector
        output = self.head(cls_token_final)  # (batch_size, 2)

        return output


class ChannelWiseTransformerNet(nn.Module):
    """Transformer with full channel-wise processing in both embedding and attention"""

    def __init__(
        self,
        input_channels,  # Number of channels/features
        input_seq_size,  # Length of input time series
        output_size,  # Output dimension
        patch_size=50,  # Size of each patch
        embedding_size=256,  # Embedding dimension
        attn_n_heads=6 * [8],  # Number of attention heads for each transformer block
        dropout=0.1,
    ):
        super().__init__()

        assert (
            embedding_size % input_channels == 0
        ), f"embedding size is not divisible by the number of channels: {embedding_size=}, {input_channels=}"

        # Channel-wise patch embedding
        self.patch_embed = ChannelWisePatchEmbedding1D(
            patch_size, embedding_size, input_channels
        )
        num_patches = input_seq_size // patch_size

        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches + 1, embedding_size) * 0.02
        )

        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_size) * 0.02)

        # Channel-wise transformer blocks
        self.transformer = nn.ModuleList(
            [
                ChannelWiseTransformerBlock(
                    embedding_size, nh, input_channels, dropout=dropout
                )
                for nh in attn_n_heads
            ]
        )

        # Final layer norm
        self.norm = nn.LayerNorm(embedding_size)

        # Classification head
        self.head = nn.Linear(embedding_size, output_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch_size, input_channels, input_seq_size)
        batch_size = x.shape[0]

        # Create patches and embed
        x = self.patch_embed(x)  # (batch_size, num_patches, embedding_size)

        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional embedding
        x = x + self.pos_embed
        x = self.dropout(x)

        # Apply channel-wise transformer blocks
        for transformer_block in self.transformer:
            x = transformer_block(x)

        # Layer norm
        x = self.norm(x)

        # Use class token for final prediction
        cls_token_final = x[:, 0]

        # Output
        output = self.head(cls_token_final)

        return output


# --------------------------------------
# Tests
# --------------------------------------

# TODO use doxygen for these test


def test_TransformerNet():
    print("---------------------------------------^")
    batch_size = 4
    input_channels = 1
    input_size = 1000
    output_size = 2

    net = TransformerNet(
        input_seq_size=input_size,
        output_size=output_size,
        patch_size=50,
        embedding_size=128,
        attn_n_heads=6 * [8],
    )
    print(net)
    print("Total number of parameters =", sum(p.numel() for p in net.parameters()))

    print("Test 1:")
    x = torch.ones((batch_size, input_channels, input_size))
    y = net(x)
    print("- input  x.size() =", x.size())
    print("- output y.size() =", y.size())
    print("- input  x =", x, sep="\n")
    print("- output y =", y, sep="\n")
    print("---------------------------------------$")


def test_ChannelWiseTransformerNet():
    print("---------------------------------------^")
    batch_size = 4
    input_channels = 3
    input_size = 1000
    output_size = 2

    net = ChannelWiseTransformerNet(
        input_channels=input_channels,
        input_seq_size=input_size,
        output_size=output_size,
        patch_size=50,
        embedding_size=192,
        attn_n_heads=6 * [8],
    )
    print(net)
    print("Total number of parameters =", sum(p.numel() for p in net.parameters()))

    print("Test 1:")
    x = torch.ones((batch_size, input_channels, input_size))
    y = net(x)
    print("- input  x.size() =", x.size())
    print("- output y.size() =", y.size())
    print("- input  x =", x, sep="\n")
    print("- output y =", y, sep="\n")
    print("---------------------------------------$")


if __name__ == "__main__":
    r"""Runs tests."""
    test_TransformerNet()
    test_ChannelWiseTransformerNet()
