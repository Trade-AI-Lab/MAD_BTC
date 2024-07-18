import torch
import sys
import os

sys.path.append(
    os.path.abspath(__file__)[
        : os.path.abspath(__file__).find("MAD_BTC") + len("MAD_BTC")
    ]
)

from model.activation_function.torch_activations import Activation
from model.nn import *
from model.activation_function.activations import *


def selecting_keys(dictionnary: dict, class_name: str):
    keys = {key: value for key, value in dictionnary.items() if str(class_name) in key}
    return keys


class MajorAutoencoder(torch.nn.Module):
    def __init__(self, config: dict, activation: str, *args, **kwargs):
        """

        Args:
            config (dict): dictionnary representing the configuration
            activation (str): string representing the activation function to be used
            *args ():
            **kwargs ():
        """

        super(MajorAutoencoder, self).__init__(*args, **kwargs)
        self.config = config
        self.activation = activation
        self.encoder = None
        self.decoder = None
        self.latent_space = None
        self.list_layers_encoder = []
        self.list_layers_decoder = []

    def builder_mlp(self, key):
        layer = torch.nn.LazyLinear(*self.config[key])
        self.x_init = layer(self.x_init)
        inverse_layer = torch.nn.LazyLinear(out_features=layer.in_features)
        return layer, inverse_layer

    def builder_cnn(self, key):
        Conv = LazyConv2d if self.x_init.dim() == 4 else LazyConv1d
        ConvTranspose = (
            LazyConvTranspose2d if self.x_init.dim() == 4 else LazyConvTranspose1d
        )
        layer = Conv(*self.config[key])
        self.x_init = layer(self.x_init)  # needed to forward iteratively
        inverse_layer = ConvTranspose(**layer.get_parameters())
        return layer, inverse_layer

    def builder_rnn(self, key):
        assert self.x_init.dim() == 3
        layer = LazyLstm(*self.config[key])
        self.x_init = layer(self.x_init)
        inverse_layer = LazyLstm(**layer.get_parameters())
        return layer, inverse_layer

    def builder(self, key):
        """
        Args:
            key (str):string representing the key used for choosing the right element in the self.config

        Example:
            layer = encoded_layer(self.config[key])
            x = layer(x)
            inverse_layer = decoded_layer(layer.in_features)

        """
        if self.config["builder"] == "mlp":
            return self.builder_mlp(key)
        elif self.config["builder"] == "cnn":
            return self.builder_cnn(key)
        elif self.config["builder"] == "rnn":
            return self.builder_rnn(key)
        else:
            raise f"Builder {self.config['builder']} is unknown."

    def loop_build(self, key):
        """
        Loop through the configuration keys to build encoder and decoder layers.

        Args:
            key (str): String representing the key used for choosing the right element in the self.config.

        Returns:
            None
        """
        layer, inverse_layer = self.builder(key)
        self.list_layers_encoder.append(layer)
        self.list_layers_encoder.append(Activation.apply_activation(self.activation)())
        self.list_layers_decoder.append(Activation.apply_activation(self.activation)())
        self.list_layers_decoder.append(inverse_layer)

    def build(self, x: torch.Tensor):
        """
        Function used before the propagation in order to compute the encoder and decoder.

        Args:
            x (torch.Tensor): A tensor to initialize the build function.

        Returns:
            None
        """
        self.x_init = x
        for key in selecting_keys(self.config, "layer"):
            self.loop_build(key)
        self.list_layers_decoder = self.list_layers_decoder[
            ::-1
        ]  # Reverse the list of layers
        self.encoder = torch.nn.Sequential(*self.list_layers_encoder)
        self.decoder = torch.nn.Sequential(*self.list_layers_decoder)

    def forward(self, x):
        """
        Forward pass through the autoencoder.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        if self.encoder is None:
            self.build(x)
        self.latent_space = self.encoder(x)
        z = self.decoder(self.latent_space)
        return z


class VariationnalMajorAutoencoder(MajorAutoencoder):

    def __init__(self, config, activation, *args, **kwargs):
        super(VariationnalMajorAutoencoder, self).__init__(
            config, activation, *args, **kwargs
        )

    def prior(self, x):
        raise "To be Implemented"

    def posterior_sample(self, *args):
        raise "To be Implemented"

    def kl_divergence(self) -> torch.Tensor:
        raise "To be Implemented"

    def build(self, x: torch.Tensor):
        """
        Function used before the propagation in order to compute the encoder and decoder.

        Args:
            x (torch.Tensor): A tensor to initialize the build function.

        Returns:
            None
        """
        self.x_init = x
        for key in selecting_keys(self.config, "layer"):
            self.loop_build(key)
        self.list_layers_decoder = self.list_layers_decoder[
            ::-1
        ]  # Reverse the list of layers
        self.encoder = torch.nn.Sequential(*self.list_layers_encoder)
        self.decoder = torch.nn.Sequential(*self.list_layers_decoder)

    def forward(self, x):
        """
        Forward pass through the autoencoder.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        if self.encoder is None:
            self.build(x)
        self.latent_space = self.encoder(x)
        z = self.decoder(self.latent_space)
        return z


class VariationnalNormalAutoencoder(VariationnalMajorAutoencoder):
    def __init__(self, config, activation, *args, **kwargs):
        super(VariationnalNormalAutoencoder, self).__init__(
            config, activation, *args, **kwargs
        )
        self.latent_size = self.config["stochastic"]["latent_size"]

    def prior(self, x: torch.Tensor) -> torch.distributions:
        self.prior = torch.distributions.Normal(torch.zeros_like(x), torch.ones_like(x))

    def posterior_sample(self, mu, logvar):
        self.posterior = torch.distributions.normal.Normal(mu, torch.exp(0.5 * logvar))
        return self.posterior.sample()

    def kl_divergence(self):
        self.kl_loss = torch.sum(
            torch.distributions.kl.kl_divergence(self.prior, self.posterior), axis=-1
        )

    def build(self, x: torch.Tensor):
        super(VariationnalNormalAutoencoder, self).build(x)
        self.list_layers_decoder.insert(
            0, torch.nn.LazyLinear(out_features=self.encoder(x).size(-1))
        )
        self.decoder = torch.nn.Sequential(*self.list_layers_decoder)
        self.list_layers_encoder.append(
            torch.nn.LazyLinear(out_features=2 * self.latent_size)
        )
        self.encoder = torch.nn.Sequential(*self.list_layers_encoder)

    def forward(self, x):
        """
        Forward pass through the autoencoder.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        if self.encoder is None:
            self.build(x)
        self.latent_space = self.encoder(x)
        mu, logvar = torch.split(self.latent_space, self.latent_size, dim=-1)
        sample = self.posterior_sample(mu, logvar)
        z = self.decoder(sample)
        return z


class VariationnalGumbelAutoencoder(VariationnalNormalAutoencoder):

    def __init__(self, config, activation, *args, **kwargs):
        super(VariationnalGumbelAutoencoder, self).__init__(
            config, activation, *args, **kwargs
        )
        self.softplus = torch.nn.Softplus()

    def prior(self, x: torch.Tensor) -> torch.distributions:
        self.prior = torch.distributions.gumbel.Gumbel(
            torch.zeros_like(x), torch.ones_like(x)
        )

    def posterior_sample(self, mu, logvar):
        self.posterior = torch.distributions.gumbel.Gumbel(mu, self.softplus(logvar))
        return self.posterior.sample()


class VariationnalWeibullAutoencoder(VariationnalNormalAutoencoder):

    def __init__(self, config, activation, *args, **kwargs):
        super(VariationnalWeibullAutoencoder, self).__init__(
            config, activation, *args, **kwargs
        )
        self.softplus = torch.nn.Softplus()

    def prior(self, x: torch.Tensor) -> torch.distributions:
        self.prior = torch.distributions.weibull.Weibull(
            torch.ones_like(x), torch.ones_like(x)
        )

    def posterior_sample(self, mu, logvar):
        self.posterior = torch.distributions.weibull.Weibull(
            self.softplus(mu), self.softplus(logvar)
        )
        return self.posterior.sample()


if __name__ == "__main__":

    config = {
        "layer_1": (32, 12, 4, 0, "zeros", 1, 1, "cpu"),
        "layer_2": (16, 2, 2, 0, "zeros", 1, 1, "cpu"),
        "builder": "cnn",
        "stochastic": {"latent_size": 24},
    }
    x = torch.randn(32, 48, 64)

    AutoEncoder = MajorAutoencoder(config=config, activation="GELU")

    z = AutoEncoder(x)
    y = AutoEncoder.encoder(x)
    print(y.size())
    print(x.size() == z.size())

    config = {
        "layer_1": (32, True),
        "layer_2": (16, True),
        "builder": "mlp",
        "stochastic": {"latent_size": 24},
    }
    x = torch.randn(32, 48, 32, 64)
    AutoEncoder = VariationnalWeibullAutoencoder(config=config, activation="GELU")
    z = AutoEncoder(x)
    y = AutoEncoder.encoder(x)
    print(y.size())
    print(x.size() == z.size())

    config = {
        "layer_1": (32, 3, True, False, 0.3, True, 0, "cpu"),
        "layer_2": (16, 3, True, False, 0.3, True, 0, "cpu"),
        "builder": "rnn",
        "stochastic": {"latent_size": 24},
    }
    x = torch.randn(32, 48, 64)
    AutoEncoder = VariationnalNormalAutoencoder(config=config, activation="GELU")
    z = AutoEncoder(x)
    y = AutoEncoder.encoder(x)
    print(y.size())
    print(x.size() == z.size())

    config = {
        "layer_1": (32, 3, True, False, 0.3, True, 0, "cpu"),
        "layer_2": (16, 3, True, False, 0.3, True, 0, "cpu"),
        "builder": "rnn",
        "stochastic": {"latent_size": 24},
    }
    x = torch.randn(32, 48, 64)
    AutoEncoder = VariationnalGumbelAutoencoder(config=config, activation="GELU")
    z = AutoEncoder(x)
    y = AutoEncoder.encoder(x)
    print(y.size())
    print(x.size() == z.size())

    config = {
        "layer_1": (32, 3, True, False, 0.3, True, 0, "cpu"),
        "layer_2": (16, 3, True, False, 0.3, True, 0, "cpu"),
        "builder": "rnn",
        "stochastic": {"latent_size": 24},
    }
    x = torch.randn(32, 48, 64)

    AutoEncoder = VariationnalWeibullAutoencoder(config=config, activation="GELU")
    z = AutoEncoder(x)
    y = AutoEncoder.encoder(x)
    print(y.size())
    print(x.size() == z.size())

    config = {"layer_1": (32, True), "layer_2": (16, True), "builder": "mlp"}
    x = torch.randn(32, 48, 64)

    AutoEncoder = MajorAutoencoder(config=config, activation="GELU")
    z = AutoEncoder(x)
    y = AutoEncoder.encoder(x)
    print(y.size())
    print(x.size() == z.size())

    config = {
        "layer_1": (32, [3, 3], [1, 1], [0, 0], "zeros", [1, 1], 1, "cpu"),
        "layer_2": (32, [3, 3], [1, 1], [0, 0], "zeros", [1, 1], 1, "cpu"),
        "builder": "cnn",
    }
    x = torch.randn(32, 48, 32, 64)
    AutoEncoder = MajorAutoencoder(config=config, activation="GELU")
    z = AutoEncoder(x)
    y = AutoEncoder.encoder(x)
    print(y.size())
    print(x.size() == z.size())

    config = {
        "layer_1": (32, 3, True, False, 0.3, True, 0, "cpu"),
        "layer_2": (16, 3, True, False, 0.3, True, 0, "cpu"),
        "builder": "rnn",
    }
    x = torch.randn(32, 48, 64)
    AutoEncoder = MajorAutoencoder(config=config, activation="GELU")
    z = AutoEncoder(x)
    y = AutoEncoder.encoder(x)
    print(y.size())
    print(x.size() == z.size())

    config = {
        "layer_1": (32, 12, 4, 0, "zeros", 1, 1, "cpu"),
        "layer_2": (16, 2, 2, 0, "zeros", 1, 1, "cpu"),
        "builder": "cnn",
        "stochastic": {"latent_size": 24},
    }
    x = torch.randn(32, 48, 64)

    AutoEncoder = VariationnalWeibullAutoencoder(config=config, activation="GELU")

    z = AutoEncoder(x)
    y = AutoEncoder.encoder(x)
    print(y.size())
    print(x.size() == z.size())

    config = {
        "layer_1": (32, [3, 3], [1, 1], [0, 0], "zeros", [1, 1], 1, "cpu"),
        "layer_2": (32, [3, 3], [1, 1], [0, 0], "zeros", [1, 1], 1, "cpu"),
        "builder": "cnn",
        "stochastic": {"latent_size": 24},
    }
    x = torch.randn(32, 48, 32, 64)
    AutoEncoder = VariationnalWeibullAutoencoder(config=config, activation="GELU")
    z = AutoEncoder(x)
    y = AutoEncoder.encoder(x)
    print(y.size())
    print(x.size() == z.size())
