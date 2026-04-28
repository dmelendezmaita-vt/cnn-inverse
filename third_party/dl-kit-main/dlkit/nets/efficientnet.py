"""
EfficientNet-type architecture for time series data.
"""

from collections import namedtuple
import math
import torch
import torch.nn as nn

# --------------------------------------
# Config
# --------------------------------------

# EfficientNet-B0 configuration
# Each tuple: (kernel_size, stride, expand_ratio, input_channels, output_channels, num_layers, se_ratio)
MBConvConfig = namedtuple(
    "MBConvConfig",
    [
        "kernel_size",
        "stride",
        "expand_ratio",
        "input_channels",
        "output_channels",
        "num_layers",
        "se_ratio",
    ],
)


def get_efficientnet_b0_config():
    """Get EfficientNet-B0 configuration adapted for 1D with ~5x fewer parameters"""
    return [
        MBConvConfig(
            kernel_size=3,
            stride=1,
            expand_ratio=1,
            input_channels=16,
            output_channels=8,
            num_layers=1,
            se_ratio=0.25,
        ),
        MBConvConfig(
            kernel_size=3,
            stride=2,
            expand_ratio=4,
            input_channels=8,
            output_channels=12,
            num_layers=1,
            se_ratio=0.25,
        ),
        MBConvConfig(
            kernel_size=5,
            stride=2,
            expand_ratio=4,
            input_channels=12,
            output_channels=20,
            num_layers=1,
            se_ratio=0.25,
        ),
        MBConvConfig(
            kernel_size=3,
            stride=2,
            expand_ratio=4,
            input_channels=20,
            output_channels=40,
            num_layers=2,
            se_ratio=0.25,
        ),
        MBConvConfig(
            kernel_size=5,
            stride=1,
            expand_ratio=4,
            input_channels=40,
            output_channels=56,
            num_layers=2,
            se_ratio=0.25,
        ),
        MBConvConfig(
            kernel_size=5,
            stride=2,
            expand_ratio=4,
            input_channels=56,
            output_channels=96,
            num_layers=2,
            se_ratio=0.25,
        ),
        MBConvConfig(
            kernel_size=3,
            stride=1,
            expand_ratio=4,
            input_channels=96,
            output_channels=160,
            num_layers=1,
            se_ratio=0.25,
        ),
    ]


# --------------------------------------
# Components
# --------------------------------------


class Swish(nn.Module):
    """Swish activation function"""

    def forward(self, x):
        return x * torch.sigmoid(x)


class SqueezeExcitation1D(nn.Module):
    """Squeeze-and-Excitation block for 1D convolutions"""

    def __init__(self, input_channels, se_ratio=0.25):
        super().__init__()
        squeeze_channels = max(1, int(input_channels * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(input_channels, squeeze_channels, 1),
            Swish(),
            nn.Conv1d(squeeze_channels, input_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)


class MBConv1D(nn.Module):
    """Mobile Inverted Bottleneck Convolution for 1D time series"""

    def __init__(self, config, dropout=0.0):
        super().__init__()
        self.config = config
        self.has_se = config.se_ratio is not None and config.se_ratio > 0
        self.use_residual = (
            config.stride == 1 and config.input_channels == config.output_channels
        )

        # Expansion phase
        expanded_channels = config.input_channels * config.expand_ratio
        if config.expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv1d(config.input_channels, expanded_channels, 1, bias=False),
                nn.BatchNorm1d(expanded_channels),
                Swish(),
            )
        else:
            self.expand_conv = nn.Identity()

        # Depthwise convolution
        self.depthwise_conv = nn.Sequential(
            nn.Conv1d(
                expanded_channels,
                expanded_channels,
                config.kernel_size,
                stride=config.stride,
                padding=config.kernel_size // 2,
                groups=expanded_channels,
                bias=False,
            ),
            nn.BatchNorm1d(expanded_channels),
            Swish(),
        )

        # Squeeze-and-Excitation
        if self.has_se:
            self.se = SqueezeExcitation1D(expanded_channels, config.se_ratio)

        # Output projection
        self.project_conv = nn.Sequential(
            nn.Conv1d(expanded_channels, config.output_channels, 1, bias=False),
            nn.BatchNorm1d(config.output_channels),
        )

        # Dropout for residual connection
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x):
        identity = x

        # Expansion
        x = self.expand_conv(x)

        # Depthwise convolution
        x = self.depthwise_conv(x)

        # Squeeze-and-Excitation
        if self.has_se:
            x = self.se(x)

        # Output projection
        x = self.project_conv(x)

        # Residual connection
        if self.use_residual:
            if self.dropout is not None:
                x = self.dropout(x)
            x = x + identity

        return x


# --------------------------------------
# EfficientNet
# --------------------------------------


class EfficientNet1D(nn.Module):
    """EfficientNet adapted for 1D time series input with 2D vector output"""

    def __init__(
        self,
        input_channels=1,
        input_length=1000,
        num_classes=2,
        dropout_connect=0.2,
        dropout_head=0.2,
    ):
        super().__init__()

        self.config = get_efficientnet_b0_config()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(16),
            Swish(),
        )

        # Mobile Inverted Bottleneck blocks
        self.blocks = nn.ModuleList()
        for stage_config in self.config:
            for i in range(stage_config.num_layers):
                # Only first block in each stage uses the specified stride
                if i == 0:
                    block_config = stage_config
                else:
                    block_config = stage_config._replace(
                        input_channels=stage_config.output_channels, stride=1
                    )

                # Stochastic depth (drop connect)
                dropout_block = (
                    dropout_connect
                    * len(self.blocks)
                    / sum(cfg.num_layers for cfg in self.config)
                )

                self.blocks.append(MBConv1D(block_config, dropout_block))
            out_channels = stage_config.output_channels

        # Head
        self.head = nn.Sequential(
            nn.Conv1d(out_channels, 640, 1, bias=False),
            nn.BatchNorm1d(640),
            Swish(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout_head),
            nn.Linear(640, num_classes),
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using standard techniques"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x shape: (batch_size, sequence_length) -> (batch_size, 1, sequence_length)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Stem
        x = self.stem(x)

        # MBConv blocks
        for block in self.blocks:
            x = block(x)

        # Head
        x = self.head(x)

        return x


# --------------------------------------
# Tests
# --------------------------------------

# TODO use doxygen for these test


# Example usage and testing
def test_efficientnet_1d():
    # Create model
    model = EfficientNet1D(input_length=1000, num_classes=2)

    # Create dummy input (batch_size=4, sequence_length=1000)
    x = torch.randn(4, 1000)

    # Forward pass
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    return model, output


if __name__ == "__main__":
    model, output = test_efficientnet_1d()
    print("Model created successfully!")
    print(f"Sample output: {output[0].detach().numpy()}")
