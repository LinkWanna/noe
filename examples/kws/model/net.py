import brevitas.nn as qnn
import torch
import torch.nn as nn
import torchinfo
from brevitas.nn.mixin.act import ActQuantType
from brevitas.nn.mixin.parameter import BiasQuantType, WeightQuantType
from brevitas.quant import Int8ActPerTensorFixedPoint, Int8WeightPerTensorFixedPoint, Int16Bias


class QuantKWS(nn.Module):
    def __init__(
        self,
        num_classes: int,
        weight_quant: WeightQuantType = Int8WeightPerTensorFixedPoint,
        act_quant: ActQuantType = Int8ActPerTensorFixedPoint,
        bias_quant: BiasQuantType = Int16Bias,
    ):
        super().__init__()

        self.features = nn.Sequential(
            qnn.QuantConv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=(5, 5),
                stride=(1, 1),
                padding=0,
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
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=0),
            qnn.QuantConv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=0,
                bias=True,
                input_quant=act_quant,
                weight_quant=weight_quant,
                bias_quant=bias_quant,
            ),
            qnn.BatchNorm2dToQuantScaleBias(
                num_features=32,
                input_quant=act_quant,
                weight_quant=weight_quant,
                bias_quant=bias_quant,
            ),
            qnn.QuantReLU(act_quant=act_quant),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=0),
            qnn.QuantConv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=0,
                bias=True,
                input_quant=act_quant,
                weight_quant=weight_quant,
                bias_quant=bias_quant,
            ),
            qnn.BatchNorm2dToQuantScaleBias(
                num_features=64,
                input_quant=act_quant,
                weight_quant=weight_quant,
                bias_quant=bias_quant,
            ),
            qnn.QuantReLU(act_quant=act_quant),
            nn.Dropout(p=0.2),
            qnn.QuantConv2d(
                in_channels=64,
                out_channels=32,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=0,
                bias=True,
                input_quant=act_quant,
                weight_quant=weight_quant,
                bias_quant=bias_quant,
            ),
            qnn.BatchNorm2dToQuantScaleBias(
                num_features=32,
                input_quant=act_quant,
                weight_quant=weight_quant,
                bias_quant=bias_quant,
            ),
            qnn.QuantReLU(act_quant=act_quant),
            nn.Dropout(p=0.3),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            qnn.QuantLinear(
                576,
                num_classes,
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
    # Example input shape for KWS spectrogram-like feature maps
    model = QuantKWS(num_classes=35)
    torchinfo.summary(model, input_size=(1, 1, 63, 12))
