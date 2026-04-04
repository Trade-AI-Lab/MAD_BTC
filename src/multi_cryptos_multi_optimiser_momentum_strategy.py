import os
import optuna
import copy
import toml
import asyncio
import pandas as pd
import imageio

from itertools import product
from src.strategy_utils.launcher_multi_strategy import multi_trading_strategy
from src.utils.utils_signature import create_signature
from src.utils.data_checks import check_data_availability_in_folder
from src.utils.downloader import downloader

absolute_path = os.path.abspath(__file__)[
    : os.path.abspath(__file__).find("MAD_BTC") + len("MAD_BTC")
]

def from_list_to_str(list_symbols:str)->str:
    symbols_sorted = sorted(list_symbols)
    str_symbols = "_".join(symbols_sorted)
    # Join the sorted elements with underscores
    return str_symbols
    
class trading_optimisation:
    """
    Trading Optimization

    This class provides a framework for optimizing trading strategies using Optuna.

    """

    def __init__(self, config: dict) -> None:
        """
        Initialize Trading Optimization

        This method initializes the trading optimization.

        :param config: Configuration parameters for optimization.
        :type config: dict
        """
        self.strategy = None
        self.config = config
        os.makedirs(absolute_path + "/study", exist_ok=True)  # create the file study
        self.liste = []

    def optimiser_multiple_currency(
        self,
        list_crypto_currencies: str = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "BNBUSDT"],
        list_time: str = ["1h", "4h", "1d"],
        list_brokers=["binance"],
    ):
        """
        Optimize Multiple Currencies

        This method performs optimization for multiple cryptocurrencies, time intervals, and brokers.

        :param list_crypto_currencies: List of cryptocurrency symbols, defaults to ["BTCUSDT", "ETHUSDT", "ADAUSDT", "BNBUSDT"]
        :type list_crypto_currencies: str, optional
        :param list_time: List of time intervals, defaults to ["1h", "4h", "1d"]
        :type list_time: str, optional
        :param list_brokers: List of broker names, defaults to ["binance"]
        :type list_brokers: list, optional
        """
        nested_list = list(product(list_time, list_brokers))
        for element in nested_list:
            time, broker = element
            self.loop_optimiser_multiple_currency(list_crypto_currencies, time, broker)

    def loop_optimiser_multiple_currency(self, list_crypto_currencies, time, broker):
        """
        Loop Optimize Multiple Currencies

        This method allows optimizing a trading strategy for a specific cryptocurrency, time interval, and broker.

        :param list_crypto_currencies: Cryptocurrency symbol.
        :type crypto_currency: str
        :param time: Time interval.
        :type time: str
        :param broker: Broker name.
        :type broker: str
        """
        
        symbols, self.config["time"], self.config["broker"] = (
            list_crypto_currencies,
            time,
            broker,
        )  # because we are looped on them

        # Join the sorted elements with underscores
        str_list_crypto_currencies = from_list_to_str(symbols)
        study_name = (
            "study"
            + f"_{self.config['strategy']['name']}_{str_list_crypto_currencies}_{self.config['exchange']}_{self.config['time']}"
        )  # the study_name depending on the strategy name, crypto,exchange and time
        storage_name = f"sqlite:///{os.path.join(absolute_path, 'study', study_name)}.db"  # storage name
        study = optuna.create_study(
            storage_name,
            sampler=optuna.samplers.TPESampler(
                multivariate=True, group=True, constant_liar=True, n_startup_trials=150
            ),
            study_name=study_name,
            direction="maximize",
            load_if_exists="True",
        )  # study created else we reload the past strategy
        study.optimize(
            lambda trial: self.objective(trial=trial), n_trials=300
        )  # we call optuna to optimize it
        self.save_df()
        self.save_mp4()

    def save_df(self):
        """
        save_df

        This method defines the save_df function for saving data into a dataframe.
        """
        raise "To be implemented"

    def save_mp4(self):
        """
        save_mp4

        This method defines the save_mp4 function for saving data into a mp4.
        """
        raise "To be implemented"

    def objective(self, trial: optuna.trial):
        """
        Objective

        This method defines the objective function for optimization.

        :param trial: Optuna trial object.
        :type trial: optuna.trial
        """
        raise "To be implemented"


class optimiser_multi_assets_momentum_strategy(trading_optimisation):
    """
    Optimizer for Momentum Strategy

    This class specializes in optimizing the Momentum strategy.

    """

    def __init__(self, config: dict) -> None:
        """
        Initialize Optimizer for Momentum Strategy

        This method initializes the optimizer for the Momentum strategy.

        :param config: Configuration parameters for optimization.
        :type config: dict
        """
        super().__init__(config)
        self.strategy = multi_trading_strategy(config)  # initialize the momentum strategy
        self.list_crypto_currencies_str = from_list_to_str(self.config['list_crypto_currencies'])
        
    def objective(self, trial: optuna.trial) -> None:
        """
        Objective for Momentum Strategy Optimization

        This method defines the objective function for optimizing the Momentum strategy.

        :param trial: Optuna trial object.
        :type trial: optuna.trial
        :return: None
        :rtype: None
        """

        self.config["strategy"]["x"] = trial.suggest_int("x", 1, 50)  # hyperparameters
        y_suggested = trial.suggest_int(
            "y", 1, 120
        )  # hyperparameters
        self.config["strategy"]["y"] = y_suggested + self.config["strategy"]["x"]
        self.config["strategy"]["ratio_method"] = trial.suggest_categorical(
            "ratio_method",
            ["ma_ratio", "ema_ratio", "zlema_ratio", "dema_ratio", "tema_ratio"],
        )  # hyperparameters
        self.strategy.change_config(
            self.config
        )  # essential change config wil reinitialize the object
        simulation_df = self.strategy.compute_simulation()  # compute the simulation
        result_portfolio = simulation_df["Portfolio"].iloc[-1]
        rsi = (simulation_df[f"RSI_Portfolio/Buy_and_Hold"]).mean()
        result = result_portfolio * rsi  # calculate a result
        config = copy.deepcopy(self.config)
        self.liste.append(
            [config, result, result_portfolio, rsi, create_signature(self.config)]
        )

        return result

    def save_df(self):
        df = pd.DataFrame(
            self.liste,
            columns=["config", "result", "result_portfolio", "rsi", "signature"],
        )
        self.dataframe_path = (
            self.strategy.time_eccefpsn
            + f"/{self.list_crypto_currencies_str}_{self.config['time']}_{self.config['strategy']['name']}.csv"
        )
        if os.path.exists(self.dataframe_path):
            existing_df = pd.read_csv(self.dataframe_path)
            df = pd.concat([df, existing_df], ignore_index=True)
            del existing_df
        df = df.drop_duplicates(subset=["result"])
        self.df = df.drop_duplicates(subset=["signature"])
        self.df.to_csv(self.dataframe_path, index=False)
        self.liste = []

    def save_mp4(self):
        mp4_file = (
            self.strategy.time_eccefpsn
            + f"_{self.list_crypto_currencies_str}_{self.config['time']}_output_video.mp4"
        )

        # Create writer object to write the video
        writer = imageio.get_writer(mp4_file, fps=2)  # Adjust the fps as needed
        # Loop through each directory in self.strategy.time_eccefpsn
        for dir_name in os.listdir(self.strategy.time_eccefpsn):
            if not dir_name.endswith(".csv"):
                dir_path = os.path.join(self.strategy.time_eccefpsn, dir_name)

                # List files in the directory
                files = os.listdir(dir_path)

                # Filter out PNG files
                png_files = [file for file in files if file.endswith(".png")]

                # Loop through PNG files in the directory
                for png_file in png_files:
                    png_full_path = os.path.join(dir_path, png_file)

                    # Read the PNG image
                    image = imageio.imread(png_full_path)

                    # Append the image to the video
                    writer.append_data(image)

        # Close the writer
        writer.close()
        print("MP4 Done")



if __name__ == "__main__":
    absolute_path = os.path.abspath(__file__)[
        : os.path.abspath(__file__).find("MAD_BTC") + len("MAD_BTC")
    ]

    # read the config file:
    with open("src/configs/optimiser_momentum_multi_cryptos_config.toml", "r") as file:
        config = toml.load(file)
    missing_data = check_data_availability_in_folder(
        "data",
        config["multi"]["list_brokers"],
        config["multi"]["list_crypto_currencies"],
        config["multi"]["list_time"],
    )
    if len(missing_data) != 0:
        print("Downloading missing data")
        with open("src/configs/download_config.toml", "r") as file:
            data_config = toml.load(file)
        data_config["symbols"] = missing_data[0]
        data_config["time_frame"] = missing_data[1]
        data_config["broker"] = missing_data[2]
        loop = asyncio.get_event_loop()
        loop.run_until_complete(downloader(data_config))
        print("Downloading finished")

    optimiser_multi_assets_momentum_strategy(config=config).optimiser_multiple_currency(
        **config["multi"]
    )
