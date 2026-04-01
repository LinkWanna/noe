import brevitas.nn as qnn
import torch
import torch.nn as nn
import torchinfo
from brevitas.nn.mixin.act import ActQuantType
from brevitas.nn.mixin.parameter import BiasQuantType, WeightQuantType
from brevitas.quant import Int8ActPerTensorFixedPoint, Int8WeightPerTensorFixedPoint, Int16Bias


class QuantGesture(nn.Module):
    """
    Input(shape=input_shape)
    Conv1D(30, kernel_size=3, strides=3) -> ReLU
    Conv1D(15, kernel_size=3, strides=3) -> ReLU
    MaxPooling1D(pool_size=3, strides=3)
    Flatten
    Dense(num_classes)
    Dropout(0.5)
    """

    def __init__(
        self,
        num_classes: int,
        weight_quant: WeightQuantType = Int8WeightPerTensorFixedPoint,
        act_quant: ActQuantType = Int8ActPerTensorFixedPoint,
        bias_quant: BiasQuantType = Int16Bias,
    ):
        super().__init__()

        self.conv1 = qnn.QuantConv1d(
            in_channels=3,
            out_channels=30,
            kernel_size=3,
            stride=3,
            padding=0,
            bias=True,
            input_quant=act_quant,
            weight_quant=weight_quant,
            bias_quant=bias_quant,
        )
        self.relu1 = qnn.QuantReLU(act_quant=act_quant)

        self.conv2 = qnn.QuantConv1d(
            in_channels=30,
            out_channels=15,
            kernel_size=3,
            stride=3,
            padding=0,
            bias=True,
            input_quant=act_quant,
            weight_quant=weight_quant,
            bias_quant=bias_quant,
        )
        self.relu2 = qnn.QuantReLU(act_quant=act_quant)

        self.pool = nn.MaxPool1d(kernel_size=3, stride=3)

        self.flatten = nn.Flatten()

        self.linear = qnn.QuantLinear(
            in_features=75,
            out_features=num_classes,
            bias=True,
            input_quant=act_quant,
            weight_quant=weight_quant,
            bias_quant=bias_quant,
        )
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = self.pool(x)

        x = self.flatten(x)

        x = self.linear(x)
        x = self.dropout(x)
        return x


if __name__ == "__main__":
    model = QuantGesture(num_classes=13)
    torchinfo.summary(model, input_size=(1, 3, 150))
