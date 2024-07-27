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


class ModuleModel(pl.LightningModule):
    def __init__(self, model, loss, lr):
        super().__init__()
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
        data_config,
        batch_size,
    ):
        super().__init__()
        train_dataset = DatasetCreator(data_config["config_train"])
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_dataset = DatasetCreator(data_config["config_validation"])
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    def get_input_data(self):
        return next(iter(self.train_loader))[0]

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader


if __name__ == "__main__":
    from pytorch_lightning.callbacks import ModelCheckpoint

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

    logger = TensorBoardLogger("lightning_logs", name="classifier_logs")
    with open(r"src\classifier_ai\classifier\configs\data_config.toml", "r") as file:
        data_config = toml.load(file)
    
    with open(r"src\classifier_ai\classifier\configs\train_config.toml", "r") as file:
        train_config = toml.load(file)
        
    datamodule = DataModule(data_config, batch_size=train_config["batch_size"])
    # Init the model 
    mixer_nn = _mixer_nn()
    input_data = datamodule.get_input_data()
    mixer_nn = mixer_nn(input_data)
    # Model complete
    train_config["input_size"] = list(input_data.size())
    with open(r"src\classifier_ai\classifier\configs\train_config.toml", "w") as file:
        toml.dump(train_config, file)
    
    module_model = ModuleModel(
        mixer_nn,
        loss=torch.nn.BCELoss(),
        lr=1e-4,
    )
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()
    # Ensure validation runs after every epoch
    checkpoint_callback = ModelCheckpoint(
    dirpath=os.path.join(logger.log_dir, "checkpoints"),  # Directory to save checkpoints
    filename='model-{epoch:02d}-{val_loss:.2f}',  # Checkpoint filename pattern
    monitor='validation_loss',  # Metric to monitor
    mode='min',  # Mode to minimize the monitored metric
    save_top_k=1,  # Save only the best model based on the monitored metric
    save_weights_only=False,
    )
    
    trainer = pl.Trainer(
        logger=logger, max_epochs=train_config["max_epoch"], val_check_interval=1.0,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(
        model=module_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        
    )
