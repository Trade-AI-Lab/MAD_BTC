import numpy as np
import torch


class HestonLayer(torch.nn.Module):
    def __init__(self, time_steps, T=0.05):
        super(HestonLayer, self).__init__()
        self.dt = torch.tensor(T / time_steps, dtype=torch.float32, requires_grad=False)
        self.sqrt_dt = torch.sqrt(self.dt)
        self.time_steps = time_steps
        # LSTM for learning parameters
        self.lstm = None
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.lin_kappa = torch.nn.LazyLinear(time_steps)
        self.lin_theta = torch.nn.LazyLinear(time_steps)
        self.lin_sigma = torch.nn.LazyLinear(time_steps)
        self.lin_r = torch.nn.LazyLinear(time_steps)
        self.lin_s = torch.nn.LazyLinear(time_steps)
        self.lin_v = torch.nn.LazyLinear(time_steps)

    def build(self, x):
        self.lstm = torch.nn.LSTM(x.size(-1), 6, batch_first=True)
        self.lstm_s_v = torch.nn.LSTM(x.size(-1), 2, batch_first=True)
        self.last_lstm = torch.nn.LSTM(
            self.time_steps, self.time_steps, batch_first=True
        )

    def forward(self, x):
        if self.lstm is None:
            self.build(x)

        lstm_out, _ = self.lstm(x)
        batch_size = x.size()[0]
        lstm_out_reshaped = lstm_out.reshape(batch_size, -1)
        # Prediction of parameters
        kappa = torch.relu(self.lin_kappa(lstm_out_reshaped))
        theta = torch.sigmoid(self.lin_theta(lstm_out_reshaped))
        sigma = torch.sigmoid(self.lin_sigma(lstm_out_reshaped))
        r = torch.sigmoid(self.lin_r(lstm_out_reshaped))

        # Initialize asset price and volatility paths
        B, time_steps, _ = x.size()
        y, _ = self.lstm_s_v(x)
        S = self.lin_s(torch.tanh(y.reshape(batch_size, -1)))
        v = self.lin_v(torch.sigmoid(y.reshape(batch_size, -1)))
        dW1 = torch.randn(B, self.time_steps) * self.sqrt_dt
        dW2 = torch.randn(B, self.time_steps) * self.sqrt_dt
        v_x = kappa * (theta - v) * self.dt + 0.25 * sigma**2 * (dW2**2 - self.dt)
        S_x = (
            S * torch.exp((r - 0.5 * v_x) * self.dt + torch.sqrt(torch.abs(v_x)) * dW1)
            + self.last_lstm(S)[0]
        )
        return S_x.unsqueeze(-1)


if __name__ == "__main__":
    # Generating random input data
    B = 64  # Batch size
    time_steps = 32
    input_size = 24

    # Create random input data x with shape (B, time_steps, input_size)
    x = torch.randn(B, time_steps, 64)

    # Initialize the HestonLayer model
    heston_model = HestonLayer(time_steps=128, T=1)

    # Forward pass with the generated input data
    S = heston_model(x)

    # Print shapes of outputs
    print(f"Asset price shape: {S.shape}")
