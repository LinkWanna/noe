import brevitas.nn as qnn
import torch
import torch.nn as nn
import torchinfo
from brevitas.nn.mixin.act import ActQuantType
from brevitas.nn.mixin.parameter import BiasQuantType, WeightQuantType
from brevitas.quant import Int8ActPerTensorFixedPoint, Int8WeightPerTensorFixedPoint, Int16Bias


class QuantLeNet(nn.Module):
    """
    Quantized LeNet for MNIST (1x28x28):
    Conv(1->6, 5x5) -> ReLU -> MaxPool(2)
    Conv(6->16, 5x5) -> ReLU -> MaxPool(2)
    Flatten -> FC(16*4*4->120) -> ReLU
            -> FC(120->84) -> ReLU
            -> FC(84->10)
    """

    def __init__(
        self,
        weight_quant: WeightQuantType = Int8WeightPerTensorFixedPoint,
        act_quant: ActQuantType = Int8ActPerTensorFixedPoint,
        bias_quant: BiasQuantType = Int16Bias,
    ):
        super().__init__()

        self.features = nn.Sequential(
            qnn.QuantConv2d(
                in_channels=1,
                out_channels=6,
                kernel_size=5,
                stride=1,
                padding=0,
                bias=True,
                input_quant=act_quant,
                weight_quant=weight_quant,
                bias_quant=bias_quant,
            ),
            qnn.QuantReLU(act_quant=act_quant),
            nn.MaxPool2d(kernel_size=2, stride=2),
            qnn.QuantConv2d(
                in_channels=6,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=0,
                bias=True,
                input_quant=act_quant,
                weight_quant=weight_quant,
                bias_quant=bias_quant,
            ),
            qnn.QuantReLU(act_quant=act_quant),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            qnn.QuantLinear(
                16 * 4 * 4,
                120,
                bias=True,
                input_quant=act_quant,
                weight_quant=weight_quant,
                bias_quant=bias_quant,
            ),
            qnn.QuantReLU(act_quant=act_quant),
            qnn.QuantLinear(
                120,
                84,
                bias=True,
                input_quant=act_quant,
                weight_quant=weight_quant,
                bias_quant=bias_quant,
            ),
            qnn.QuantReLU(act_quant=act_quant),
            qnn.QuantLinear(
                84,
                10,
                bias=True,
                input_quant=act_quant,
                weight_quant=weight_quant,
                bias_quant=bias_quant,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = QuantLeNet()
    torchinfo.summary(model, input_size=(1, 1, 28, 28))
