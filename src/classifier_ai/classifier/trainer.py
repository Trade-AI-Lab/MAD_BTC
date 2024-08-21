import pytorch_lightning as pl
import torch
import sys
import os
import toml
import shutil
from sklearn.metrics import f1_score,precision_score
sys.path.append(
    os.path.abspath(__file__)[
        : os.path.abspath(__file__).find("MAD_BTC") + len("MAD_BTC")
    ]
)
from src.classifier_ai.classifier.sequencer_classifier import DatasetCreator
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from model.tsmixer import MixerLayer


class ModuleModel(pl.LightningModule):
    def __init__(self, model, loss, lr):
        super().__init__()
        self.model = model
        self.loss = loss
        self.lr = lr
        self.training_step_outputs = []   # save outputs in each batch to compute metric overall epoch
        self.training_step_targets = []   # save targets in each batch to compute metric overall epoch
        self.val_step_outputs = []        # save outputs in each batch to compute metric overall epoch
        self.val_step_targets = []        # save targets in each batch to compute metric overall epoch
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self.model(x)
        loss = self.loss(z, y)
        self.log("train_loss", loss)
        # GET AND SAVE OUTPUTS AND TARGETS PER BATCH
        y_pred = z.argmax(dim=1).cpu().numpy()
        y_true = y.cpu().numpy()
        
        # --> HERE STEP 2 <--
        self.training_step_outputs.extend(y_pred)
        self.training_step_targets.extend(y_true)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        z = self.model(x)
        loss = self.loss(z, y)
        self.log("validation_loss", loss)
        y_pred = z.argmax(dim=1).cpu().numpy()
        y_true = y.cpu().numpy()
        
        # --> HERE STEP 2 <--
        self.val_step_outputs.extend(y_pred)
        self.val_step_targets.extend(y_true)
        return loss
    def on_train_epoch_end(self):
        ## F1 Macro all epoch saving outputs and target per batch
        train_all_outputs = self.training_step_outputs
        train_all_targets = self.training_step_targets
        f1_macro_epoch = f1_score(train_all_outputs, train_all_targets, average='macro')
        precision_score_epoch = precision_score(train_all_outputs, train_all_targets, average='macro')
        self.log("training_f1_epoch", f1_macro_epoch, on_step=False, on_epoch=True, prog_bar=True)
        self.log("precision_score_epoch", precision_score_epoch, on_step=False, on_epoch=True, prog_bar=True)


        # free up the memory
        # --> HERE STEP 3 <--
        self.training_step_outputs.clear()
        self.training_step_targets.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def on_validation_epoch_end(self):
        ## F1 Macro all epoch saving outputs and target per batch
        val_all_outputs = self.val_step_outputs
        val_all_targets = self.val_step_targets

        val_f1_macro_epoch = f1_score(val_all_outputs, val_all_targets, average='macro')
        val_precision_score_epoch = precision_score(val_all_outputs, val_all_targets, average='macro')
        self.log("val_f1_epoch", val_f1_macro_epoch, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_precision_score_epoch", val_precision_score_epoch, on_step=False, on_epoch=True, prog_bar=True)


        # free up the memory
        # --> HERE STEP 3 <--
        self.val_step_outputs.clear()
        self.val_step_targets.clear()

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
    path_data_config = r"src\classifier_ai\classifier\configs\data_config.toml"
    path_train_config = r"src\classifier_ai\classifier\configs\train_config.toml"
    
    
    with open(path_data_config, "r") as file:
        data_config = toml.load(file)
    
    with open(path_train_config, "r") as file:
        train_config = toml.load(file)
    logger = TensorBoardLogger("lightning_logs", name=f"{data_config['config_train']['crypto_currency']}_{data_config['config_train']['exchange']}_{data_config['config_train']['time']}")
    datamodule = DataModule(data_config, batch_size=train_config["batch_size"])
    # Init the model 
    mixer_nn = _mixer_nn()
    input_data = datamodule.get_input_data()
    _ = mixer_nn(input_data)
    # Model complete
    train_config["input_size"] = list(input_data.size())
    with open(path_train_config, "w") as file:
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
    shutil.copy(path_data_config, logger.log_dir)
    shutil.copy(path_train_config, logger.log_dir)
