import pytorch_lightning as pl
import torch
import sys
import os
import toml
import shutil
from sklearn.metrics import f1_score,precision_score
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
sys.path.append(
    os.path.abspath(__file__)[
        : os.path.abspath(__file__).find("MAD_BTC") + len("MAD_BTC")
    ]
)
from src.classifier_ai.classifier.sequencer_classifier import DatasetCreator
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



import torch.nn.functional as F
import torch.nn as nn
class Frets(nn.Module):
    """
    Frets https://arxiv.org/pdf/2311.06184

    Frequency-domain MLPs are More Effective
    Learners in Time Series Forecasting
    """

    def __init__(
        self,
        embed_size: int = 20,
        hidden_size: int = 20,
        pred_len: int = 20,
        channel_independence: int = True,
    ):
        super(Frets, self).__init__()
        self.embed_size = embed_size  # embed_size
        self.hidden_size = hidden_size  # hidden_size
        self.pre_length = pred_len # prediction length
        self.in_features = None  # channels
        self.time_length = None # time_length
        self.channel_independence = channel_independence
        self.sparsity_threshold = 0.01
        self.scale = 0.02
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))
        self.r1 = nn.Parameter(
            self.scale * torch.randn(self.embed_size, self.embed_size)
        )
        self.i1 = nn.Parameter(
            self.scale * torch.randn(self.embed_size, self.embed_size)
        )
        self.rb1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.r2 = nn.Parameter(
            self.scale * torch.randn(self.embed_size, self.embed_size)
        )
        self.i2 = nn.Parameter(
            self.scale * torch.randn(self.embed_size, self.embed_size)
        )
        self.rb2 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib2 = nn.Parameter(self.scale * torch.randn(self.embed_size))

        self.fc = nn.Sequential(
            nn.LazyLinear(self.hidden_size),
            nn.SiLU(),
            nn.LazyLinear(self.pre_length),
        )

    # dimension extension
    def tokenEmb(self, x):
        # x: [Batch, Input length, Channel]
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(3)
        # N*T*1 x 1*D = N*T*D
        y = self.embeddings
        return x * y

    # frequency temporal learner
    def MLP_temporal(self, x, B, N, L):
        # [B, N, T, D]
        x = torch.fft.rfft(x, dim=2, norm="ortho")  # FFT on L dimension
        y = self.FreMLP(B, N, L, x, self.r2, self.i2, self.rb2, self.ib2)
        x = torch.fft.irfft(y, n=self.time_length, dim=2, norm="ortho")
        return x

    # frequency channel learner
    def MLP_channel(self, x, B, N, L):
        # [B, N, T, D]
        x = x.permute(0, 2, 1, 3)
        # [B, T, N, D]
        x = torch.fft.rfft(x, dim=2, norm="ortho")  # FFT on N dimension
        y = self.FreMLP(B, L, N, x, self.r1, self.i1, self.rb1, self.ib1)
        x = torch.fft.irfft(y, n=self.in_features, dim=2, norm="ortho")
        x = x.permute(0, 2, 1, 3)
        # [B, N, T, D]
        return x

    # frequency-domain MLPs
    # dimension: FFT along the dimension, r: the real part of weights, i: the imaginary part of weights
    # rb: the real part of bias, ib: the imaginary part of bias
    def FreMLP(self, B, nd, dimension, x, r, i, rb, ib):
        o1_real = torch.zeros(
            [B, nd, dimension // 2 + 1, self.embed_size],
        )
        o1_imag = torch.zeros(
            [B, nd, dimension // 2 + 1, self.embed_size],
        )

        o1_real = F.relu(
            torch.einsum("bijd,dd->bijd", x.real, r)
            - torch.einsum("bijd,dd->bijd", x.imag, i)
            + rb
        )

        o1_imag = F.relu(
            torch.einsum("bijd,dd->bijd", x.imag, r)
            + torch.einsum("bijd,dd->bijd", x.real, i)
            + ib
        )

        y = torch.stack([o1_real, o1_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)
        y = torch.view_as_complex(y)
        return y

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        B, T, N = x.shape
        # embedding x: [B, N, T, D]
        x = self.tokenEmb(x)
        bias = x
        if self.in_features is None:
            self.in_features = N
        if self.time_length is None:
            self.time_length = T
        # [B, N, T, D]
        if self.channel_independence:
            x = self.MLP_channel(x, B, N, T)
        # [B, N, T, D]
        x = self.MLP_temporal(x, B, N, T)
        x = x + bias
        x = self.fc(x.reshape(B, N, -1)).permute(0, 2, 1)
        # x: [Batch, Input length, pre_length]
        return x
if __name__ == "__main__":
    from pytorch_lightning.callbacks import ModelCheckpoint

    class _mixer_nn(torch.nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.mixer_model = torch.nn.Sequential(
                *[
                    Frets()
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
