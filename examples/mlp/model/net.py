import brevitas.nn as qnn
import torch.nn as nn
import torchinfo
from brevitas.nn.mixin.act import ActQuantType
from brevitas.nn.mixin.parameter import BiasQuantType, WeightQuantType
from brevitas.quant import Int8ActPerTensorFixedPoint, Int8WeightPerTensorFixedPoint, Int16Bias


class QuantMLP(nn.Module):
    """
    A simple quantized MLP for MNIST:
    28x28 -> 256 -> 256 -> 10
    """

    def __init__(
        self,
        weight_quant: WeightQuantType = Int8WeightPerTensorFixedPoint,
        act_quant: ActQuantType = Int8ActPerTensorFixedPoint,
        bias_quant: BiasQuantType = Int16Bias,
    ):
        super().__init__()
        self.features = nn.Sequential(
            nn.Flatten(),
            # Quantized Linear + Quantized ReLU
            qnn.QuantLinear(
                28 * 28,
                256,
                bias=True,
                input_quant=act_quant,
                weight_quant=weight_quant,
                bias_quant=bias_quant,
            ),
            qnn.QuantReLU(act_quant=act_quant),
            qnn.QuantLinear(
                256,
                256,
                bias=True,
                input_quant=act_quant,
                weight_quant=weight_quant,
                bias_quant=bias_quant,
            ),
            qnn.QuantReLU(act_quant=act_quant),
            # Output logits
            qnn.QuantLinear(
                256,
                10,
                bias=True,
                input_quant=act_quant,
                weight_quant=weight_quant,
                bias_quant=bias_quant,
            ),
        )

    def forward(self, x):
        return self.features(x)


if __name__ == "__main__":
    model = QuantMLP()
    torchinfo.summary(model, input_size=(1, 1, 28, 28))
