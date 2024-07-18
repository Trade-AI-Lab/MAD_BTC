import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class DainLayer(nn.Module):
    def __init__(self, mode="adaptive_avg", mode_log=False):
        super(DainLayer, self).__init__()
        self.mode = mode
        self.mode_log = mode_log
        self.input_shape = None

    def build(self, input_dim):
        """
        Thanks to build function the user does not need anymore to specify the input shape. Layers will be automatically
        computed without information.
        Returns:

        """
        self.weights = nn.Linear(input_dim * 2, 2, bias=False)

        # Parameters for adaptive average
        self.mean_layer = nn.Linear(input_dim, input_dim, bias=False)
        self.mean_layer.weight.data = torch.FloatTensor(
            data=np.eye(input_dim, input_dim)
        )

        # Parameters for adaptive std
        self.scaling_layer = nn.Linear(input_dim, input_dim, bias=False)
        self.scaling_layer.weight.data = torch.FloatTensor(
            data=np.eye(input_dim, input_dim)
        )

        # Parameters for adaptive scaling
        self.gating_layer = nn.Linear(input_dim, input_dim)

        self.eps = 1e-8

    def log_normalize(self, x):
        log_return = torch.log(torch.divide(x, x[:, 0:1, :]))
        nan_mask = torch.isnan(log_return)
        self.features_with_nan = torch.nonzero(nan_mask, as_tuple=True)[-1]
        log_return_without_nan = torch.where(
            nan_mask, torch.zeros_like(log_return), log_return
        )
        return log_return_without_nan

    def dain_normalize(self, x):
        # Nothing to normalize
        if self.mode == None:
            pass

        # Do simple average normalization
        elif self.mode == "avg":
            avg = torch.mean(x, -1)
            avg = torch.unsqueeze(avg, dim=-1)
            x = x - avg

        # Perform only the first step (adaptive averaging)
        elif self.mode == "adaptive_avg":
            avg = torch.mean(x, -1)
            adaptive_avg = self.mean_layer(avg)
            adaptive_avg = torch.unsqueeze(adaptive_avg, dim=-1)
            x = x - adaptive_avg

        # Perform the first + second step (adaptive averaging + adaptive scaling )
        elif self.mode == "adaptive_scale":

            # Step 1:
            avg = torch.mean(x, -1)
            adaptive_avg = self.mean_layer(avg)
            adaptive_avg = torch.unsqueeze(adaptive_avg, dim=-1)
            x = x - adaptive_avg

            # Step 2:
            std = torch.mean(x**2, -1)
            std = torch.sqrt(std + self.eps)
            adaptive_std = self.scaling_layer(std)
            adaptive_std[adaptive_std <= self.eps] = 1

            adaptive_std = torch.unsqueeze(adaptive_std, dim=-1)
            x = x / (adaptive_std)

        elif self.mode == "full":

            # Step 1:
            avg = torch.mean(x, -1)
            adaptive_avg = self.mean_layer(avg)
            adaptive_avg = torch.unsqueeze(adaptive_avg, dim=-1)
            x = x - adaptive_avg

            # # Step 2:
            std = torch.mean(x**2, -1)
            std = torch.sqrt(std + self.eps)
            adaptive_std = self.scaling_layer(std)
            adaptive_std[adaptive_std <= self.eps] = 1

            adaptive_std = torch.unsqueeze(adaptive_std, dim=-1)
            x = x / adaptive_std

            # Step 3:
            avg = torch.mean(x, -1)
            gate = F.sigmoid(self.gating_layer(avg))
            gate = torch.unsqueeze(gate, dim=-1)
            x = x * gate

        else:
            assert False

        return x

    @staticmethod
    def _hyperparameter():
        {
            "mode": ["avg", "adaptive_avg", "adaptive_scale", "full"],
            "mode_log": [True, False],
        }

    def forward(self, x):
        # x <=> (batch_size,time_steps,features)
        # x <=> (batch_size,cryptos,time_steps,features) => reshape into (batch_size,time_steps,features*cryptos)
        if self.input_shape is None:
            self.input_shape = x.size()
            self.build(x.size(-2))
        if self.mode_log:
            # Transpose the log-normalized and DAIN-normalized tensors to switch dimensions 1 and 2
            log_normalized = torch.transpose(self.log_normalize(x), dim0=2, dim1=1)
            dain_normalized = torch.transpose(self.dain_normalize(x), dim0=2, dim1=1)

            # Concatenate the transposed tensors along the last dimension to combine them
            combined_tensors = torch.cat((log_normalized, dain_normalized), dim=-1)

            # Apply softmax to obtain weights along dimension 1 after transposition
            weights = torch.transpose(
                torch.softmax(self.weights(combined_tensors), dim=-2), dim0=2, dim1=1
            )

            # Check if there are features with NaN values and update the weights accordingly
            if self.features_with_nan.numel() > 0:
                weights[:, 0, self.features_with_nan] = 0  # alpha
                weights[:, 1, self.features_with_nan] = 1  # beta

            # Compute the final output by applying the calculated weights
            output = weights[:, 0, :].unsqueeze(1) * self.log_normalize(x) + weights[
                :, 1, :
            ].unsqueeze(1) * self.dain_normalize(x)
        else:
            output = self.dain_normalize(x)

        # Return the final result
        return output


class DainLayerBothAxis(nn.Module):
    def __init__(
        self,
        mode_1="adaptive_avg",
        mode_log_1=False,
        mode_2="adaptive_avg",
        mode_log_2=False,
    ):
        super(DainLayerBothAxis, self).__init__()
        self.dain_x = DainLayer(mode=mode_1, mode_log=mode_log_1)
        self.dain_y = DainLayer(mode=mode_2, mode_log=mode_log_2)
        self.input_shape = None

    @staticmethod
    def _hyperparameter():
        {
            "mode_1": ["avg", "adaptive_avg", "adaptive_scale", "full"],
            "mode_2": ["avg", "adaptive_avg", "adaptive_scale", "full"],
            "mode_log_1": [True, False],
            "mode_log_2": [True, False],
        }

    def forward(self, x):
        if self.input_shape is None:
            self.dain_x.build(x.size(-2))
            self.dain_y.build(x.size(-1))
        x_1 = self.dain_x(x)
        y_1 = torch.transpose(
            self.dain_y(torch.transpose(x, dim0=2, dim1=1)), dim0=2, dim1=1
        )
        x = x_1 + y_1
        return x


if __name__ == "__main__":
    input_data = torch.randn(48, 16, 64)  # Replace with appropriate input size
    # Create an instance of the DAIN_Layer with 'full' mode
    full_dain_layer = DainLayer(mode="full", mode_log=True)
    print(full_dain_layer(input_data).size())
    dain_layer_both_axis = DainLayerBothAxis(mode_log_1=True, mode_log_2=True)
    print(dain_layer_both_axis(input_data).size())
