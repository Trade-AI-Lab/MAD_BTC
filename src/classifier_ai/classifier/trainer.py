import pytorch_lightning as pl
import torch
import sys
import os

sys.path.append(
    os.path.abspath(__file__)[
        : os.path.abspath(__file__).find("MAD_BTC") + len("MAD_BTC")
    ]
)
from src.classifier_ai.classifier.sequencer_classifier import DatasetCreator
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from src.model.tsmixer import MixerLayer
from src.model.layers.normalization_layer.dain import DainLayer


class ModuleModel(pl.LightningModule):
    def __init__(self, model, loss, input_data, lr):
        super().__init__()
        _ = model(input_data)
        self.model = model
        self.loss = loss
        self.lr = lr

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self.model(x)
        loss = self.loss(z, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        z = self.model(x)
        loss = self.loss(z, y)
        self.log("validation_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        config_train,
        config_validation,
        batch_size,
    ):
        super().__init__()
        train_dataset = DatasetCreator(config_train)
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_dataset = DatasetCreator(config_validation)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    def get_input_data(self):
        return next(iter(self.train_loader))[0]

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader


if __name__ == "__main__":
    mixer_model = torch.nn.Sequential(
        *[
            MixerLayer(
                activation=torch.nn.Tanh(),
                dropout=0.5,
                layernorm=DainLayer(mode="full", mode_log=False),
            )
            for i in range(10)
        ]
    )
    future_time_steps = 32
    mlp_model = torch.nn.Sequential()

    combined_model = torch.nn.Sequential(mixer_model, mlp_model)
    logger = TensorBoardLogger(
        "lighting_logs_try", name="conv_autoencode_predictor_logs"
    )
    price_dm = DataModule(future_time_steps=future_time_steps)
    module_model = ModuleModel(
        combined_model,
        input_data=price_dm.get_input_data(),
        loss=torch.nn.BCELoss(),
        lr=1e-4,
    )
    train_dataloader = price_dm.train_dataloader()
    val_dataloader = price_dm.val_dataloader()
    trainer = pl.Trainer(
        logger=logger, max_epochs=10, val_check_interval=1.0
    )  # Ensure validation runs after every epoch

    trainer.fit(
        model=module_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
