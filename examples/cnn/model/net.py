import brevitas.nn as qnn
import torch
import torch.nn as nn
import torchinfo
from brevitas.nn.mixin.act import ActQuantType
from brevitas.nn.mixin.parameter import BiasQuantType, WeightQuantType
from brevitas.quant import Int8ActPerTensorFixedPoint, Int8WeightPerTensorFixedPoint, Int16Bias


class QuantResidualBlock(nn.Module):
    """
    Classic residual block (no downsample):
        y = ReLU( F(x) + x )
    where
        F(x): Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN
    """

    def __init__(
        self,
        channels: int,
        weight_quant: WeightQuantType,
        act_quant: ActQuantType,
        bias_quant: BiasQuantType,
    ):
        super().__init__()

        self.conv1 = qnn.QuantConv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            input_quant=act_quant,
            weight_quant=weight_quant,
            bias_quant=bias_quant,
        )
        self.bn1 = qnn.BatchNorm2dToQuantScaleBias(
            num_features=channels,
            input_quant=act_quant,
            weight_quant=weight_quant,
            bias_quant=bias_quant,
        )
        self.relu1 = qnn.QuantReLU(act_quant=act_quant)

        self.conv2 = qnn.QuantConv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            input_quant=act_quant,
            weight_quant=weight_quant,
            bias_quant=bias_quant,
        )
        self.bn2 = qnn.BatchNorm2dToQuantScaleBias(
            num_features=channels,
            input_quant=act_quant,
            weight_quant=weight_quant,
            bias_quant=bias_quant,
        )
        self.add = qnn.QuantEltwiseAdd(input_quant=act_quant)
        self.out_relu = qnn.QuantReLU(act_quant=act_quant)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Residual add
        out = self.add(out, identity)
        out = self.out_relu(out)
        return out


class QuantCNN(nn.Module):
    """
    MNIST (1x28x28), focused on testing:
    - Conv2d with padding
    - BatchNorm
    - Classic residual connection (elementwise add)

    Stem:
        Conv(1->8, 3x3, s=1, p=1) -> BN -> ReLU -> MaxPool(2)      # 28 -> 14
        Conv(8->16, 3x3, s=1, p=1) -> BN -> ReLU -> MaxPool(2)     # 14 -> 7
        Conv(16->24, 3x3, s=1, p=1) -> BN -> ReLU                  # 7 -> 7

    Residual:
        BasicBlock(24 channels, stride=1, identity shortcut)       # 7 -> 7

    Head:
        Flatten -> FC(24*7*7 -> 64) -> ReLU -> FC(64 -> 10)
    """

    def __init__(
        self,
        weight_quant: WeightQuantType = Int8WeightPerTensorFixedPoint,
        act_quant: ActQuantType = Int8ActPerTensorFixedPoint,
        bias_quant: BiasQuantType = Int16Bias,
    ):
        super().__init__()

        self.stem = nn.Sequential(
            qnn.QuantConv2d(
                in_channels=1,
                out_channels=8,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
                input_quant=act_quant,
                weight_quant=weight_quant,
                bias_quant=bias_quant,
            ),
            qnn.BatchNorm2dToQuantScaleBias(
                num_features=8,
                input_quant=act_quant,
                weight_quant=weight_quant,
                bias_quant=bias_quant,
            ),
            qnn.QuantReLU(act_quant=act_quant),
            nn.MaxPool2d(kernel_size=2, stride=2),
            qnn.QuantConv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
                input_quant=act_quant,
                weight_quant=weight_quant,
                bias_quant=bias_quant,
            ),
            qnn.BatchNorm2dToQuantScaleBias(
                num_features=16,
                input_quant=act_quant,
                weight_quant=weight_quant,
                bias_quant=bias_quant,
            ),
            qnn.QuantReLU(act_quant=act_quant),
            nn.MaxPool2d(kernel_size=2, stride=2),
            qnn.QuantConv2d(
                in_channels=16,
                out_channels=24,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
                input_quant=act_quant,
                weight_quant=weight_quant,
                bias_quant=bias_quant,
            ),
            qnn.BatchNorm2dToQuantScaleBias(
                num_features=24,
                input_quant=act_quant,
                weight_quant=weight_quant,
                bias_quant=bias_quant,
            ),
            qnn.QuantReLU(act_quant=act_quant),
        )

        self.res_block = QuantResidualBlock(
            channels=24,
            weight_quant=weight_quant,
            act_quant=act_quant,
            bias_quant=bias_quant,
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            qnn.QuantLinear(
                24 * 7 * 7,
                64,
                bias=True,
                input_quant=act_quant,
                weight_quant=weight_quant,
                bias_quant=bias_quant,
            ),
            qnn.QuantReLU(act_quant=act_quant),
            qnn.QuantLinear(
                64,
                10,
                bias=True,
                input_quant=act_quant,
                weight_quant=weight_quant,
                bias_quant=bias_quant,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.res_block(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = QuantCNN()
    torchinfo.summary(model, input_size=(1, 1, 28, 28))
