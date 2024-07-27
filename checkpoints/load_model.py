import pytorch_lightning as pl
import torch
import sys
import os
import toml

sys.path.append(
    os.path.abspath(__file__)[
        : os.path.abspath(__file__).find("MAD_BTC") + len("MAD_BTC")
    ]
)
from src.classifier_ai.classifier.sequencer_classifier import DatasetCreator
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from model.nn import DainLayer,DainLayerBothAxis
from model.tsmixer import MixerLayer

model = torch.load("checkpoints\model-epoch=48-val_loss=0.00.ckpt")
class _mixer_nn(torch.nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.mixer_model = torch.nn.Sequential(
                *[
                    MixerLayer(
                        activation="Tanh",
                        dropout=0.5,
                        layernorm=None,
                    )
                    for i in range(10)
                ]
            )
            self.lazy_layer_1 = torch.nn.Sequential(
                torch.nn.LazyLinear(1)
            )
            self.lazy_layer_2 = torch.nn.Sequential(
                torch.nn.LazyLinear(1), torch.nn.Sigmoid()
            )

        def forward(self, input):
            x = self.mixer_model(input)  # B,C,H
            x = self.lazy_layer_1(x)  # B,C,1
            x = self.lazy_layer_2(x.squeeze(-1))  # B,1
            return x
mixer = _mixer_nn()
mixer()