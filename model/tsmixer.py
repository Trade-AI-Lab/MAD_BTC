import torch
import sys
import os

sys.path.append(
    os.path.abspath(__file__)[
        : os.path.abspath(__file__).find("MAD_BTC") + len("MAD_BTC")
    ]
)
from model.nn import DainLayer
from model.nn import Activation


class MixerLayer(torch.nn.Module):
    """
    https://arxiv.org/pdf/2303.06053.pdf
    """

    def __init__(
        self, activation: str, dropout: float, layernorm=None, *args, **kwargs
    ):
        """

        Args:
          activation (str): string representing the activation function to be used
          dropout (float): value between 0 and 1 the dropout coefficient
          *args ():
          **kwargs ():
        """
        super(MixerLayer, self).__init__(*args, **kwargs)
        self.activation_1 = Activation.apply_activation(activation)()
        self.activation_2 = Activation.apply_activation(activation)()
        self.dropout = dropout
        self.layernorm = layernorm
        self.input_shape = None
        self.dense_temporal_projection = None
        self.dropout = torch.nn.Dropout(p=dropout)
        self.dense_time_mixing = None
        self.dense_feature_mixing = None

    def build(self):
        """
        Thanks to build function the user does not need anymore to specify the input shape. Layers will be automatically
        computed without information.
        Returns:

        """
        if self.layernorm is None:
            self.layernorm = torch.nn.LayerNorm(
                [self.input_shape[-2], self.input_shape[-1]]
            )
        self.dense_temporal_projection = torch.nn.Linear(
            in_features=self.input_shape[-2], out_features=self.input_shape[-2]
        )
        self.dense_feature_mixing = torch.nn.Linear(
            in_features=self.input_shape[-1], out_features=self.input_shape[-1]
        )
        self.dense_time_mixing = torch.nn.Linear(
            in_features=self.input_shape[-1], out_features=self.input_shape[-1]
        )

    def forward(self, inputs, *args, **kwargs):
        if self.input_shape is None:
            self.input_shape = inputs.size()
            self.build()
        # Time Mixing
        x = self.layernorm(inputs)
        x = torch.transpose(x, dim0=2, dim1=1)  # [Batch, Channel, Input Length]
        x = self.dense_temporal_projection(x)
        x = self.activation_1(x)
        x = self.dropout(x)
        x = torch.transpose(x, dim0=2, dim1=1)  # [Batch, Input Length, Channel]
        res = x + inputs

        # Feature Mixing
        x = self.layernorm(res)
        x = self.dense_time_mixing(x)  # [Batch, Input Length, FF_Dim]
        x = self.activation_2(x)
        x = self.dropout(x)
        x = self.dense_feature_mixing(x)  # [Batch, Input Length, Channel]
        x = self.dropout(x)
        return x + res


class DenseLinear(torch.nn.Module):
    def __init__(self, out_features: int, device=None, dtype=None) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.out_features = out_features
        self.input_shape = None

    def build(self):
        """
        Thanks to build function the user does not need anymore to specify the input shape. Layers will be automatically
        computed without information.
        Returns:

        """
        self.dense_layer = torch.nn.Linear(
            in_features=self.input_shape[-1], out_features=self.out_features
        )

    def forward(self, inputs, *args, **kwargs):
        if self.input_shape is None:
            self.input_shape = inputs.size()
            self.build()
        # Time Mixing
        return self.dense_layer(inputs)


if __name__ == "__main__":
    input_data = torch.randn(16, 32, 64)
    print(
        MixerLayer(activation="ELU", dropout=0.3, layernorm=DainLayer(mode="full"))(
            input_data
        ).size()
    )
    print(
        MixerLayer(activation="SELU", dropout=0.3, layernorm=DainLayer(mode="full"))(
            input_data
        ).size()
    )
