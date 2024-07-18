import torch
import sys
import os

sys.path.append(
    os.path.abspath(__file__)[
        : os.path.abspath(__file__).find("MAD_BTC") + len("MAD_BTC")
    ]
)

import model.activation_function.activations as activations
from model.activation_function.activations import T_alpha


class Activation(torch.nn.Module):
    def __init__(self, func_name, boolean: True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.activation_layer = (
            T_alpha(self.apply_activation(func_name)())
            if boolean
            else self.apply_activation(func_name)()
        )
        # idea to add T_alpha https://www.sciencedirect.com/science/article/pii/S266665102300030X

    def forward(self, x):
        return self.activation_layer(x)

    @staticmethod
    def apply_activation(func_name):
        if hasattr(torch.nn, func_name):
            return getattr(torch.nn, func_name)
        elif hasattr(activations, func_name):
            return getattr(activations, func_name)
        else:
            raise ValueError(f"{func_name} is not a valid torch.nn or activations.")

    @staticmethod
    def _hyperparameters():
        return {
            "func_name": ["Symlog", "GELU", "Tanh", "Sigmoid", "Softmax", "Identity"],
            "boolean": [True, False],
        }


if __name__ == "__main__":
    print(Activation.apply_activation("Symlog")())
    Activation.apply_activation("GELU")
    Activation.apply_activation("Identity")
    Activation.apply_activation("Symlog")
    tensor = torch.randn(32, 48, 64)
    z = Activation("GELU", True)
    print(z(tensor))
