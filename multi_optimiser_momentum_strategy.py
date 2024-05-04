import pandas as pd
from src.utils.utils_df import open_dataframe
from itertools import product
from src.strategy_utils.launcher_strategy import momentum_strategy, envelop_strategy
from src.strategy_utils.strategy_comparator import evalutation
from src.utils.utils_signature import create_signature
import os
import optuna
import copy
absolute_path = os.path.abspath(__file__)[:os.path.abspath(__file__).find("MAD_BTC")+len("MAD_BTC")]
import numpy as np
class trading_optimisation:
    """
    Trading Optimization

    This class provides a framework for optimizing trading strategies using Optuna.

    """
    def __init__(self,config:dict) -> None:
        """
        Initialize Trading Optimization

        This method initializes the trading optimization.

        :param config: Configuration parameters for optimization.
        :type config: dict
        """
        self.strategy = None
        self.config = config
        os.makedirs(absolute_path+"/study",exist_ok=True) # create the file study
        self.liste = []

    
    def optimiser_multiple_currency(self,list_crypto_currency:str=["BTCUSDT","ETHUSDT","ADAUSDT","BNBUSDT"],list_time:str=["1h","4h","1d"],list_brokers=["binance"]):
        """
        Optimize Multiple Currencies

        This method performs optimization for multiple cryptocurrencies, time intervals, and brokers.

        :param list_crypto_currency: List of cryptocurrency symbols, defaults to ["BTCUSDT", "ETHUSDT", "ADAUSDT", "BNBUSDT"]
        :type list_crypto_currency: str, optional
        :param list_time: List of time intervals, defaults to ["1h", "4h", "1d"]
        :type list_time: str, optional
        :param list_brokers: List of broker names, defaults to ["binance"]
        :type list_brokers: list, optional
        """
        nested_list = list(product(list_crypto_currency, list_time, list_brokers))
        for element in nested_list:
            crypto_currency,time,broker = element
            self.loop_optimiser_multiple_currency(crypto_currency,time,broker)
                
    def loop_optimiser_multiple_currency(self,crypto_currency,time,broker):
        """
        Loop Optimize Multiple Currencies

        This method allows optimizing a trading strategy for a specific cryptocurrency, time interval, and broker.

        :param crypto_currency: Cryptocurrency symbol.
        :type crypto_currency: str
        :param time: Time interval.
        :type time: str
        :param broker: Broker name.
        :type broker: str
        """
        self.config["crypto_currency"],self.config["time"],self.config["broker"] = crypto_currency, time, broker  # because we are looped on them
        study_name = "study"+f"_{self.config['strategy']['name']}_{self.config['crypto_currency']}_{self.config['exchange']}_{self.config['time']}" # the study_name depending on the strategy name, crypto,exchange and time
        storage_name = f"sqlite:///{os.path.join(absolute_path, 'study', study_name)}.db" # storage name 
        study = optuna.create_study(storage_name, sampler=optuna.samplers.TPESampler(
            multivariate = True,
            group = True,
        constant_liar= True,
        n_startup_trials=100
        ),study_name=study_name,direction="maximize",load_if_exists="True") # study created else we reload the past strategy
        study.optimize(lambda trial: self.objective(trial=trial), n_trials=300) # we call optuna to optimize it
        self.save_df()
  
    def save_df(self):
        """
        save_df

        This method defines the save_df function for saving data into a dataframe.
        """
        raise "To be implemented"


    
    
    def objective(self,trial:optuna.trial):
        """
        Objective

        This method defines the objective function for optimization.

        :param trial: Optuna trial object.
        :type trial: optuna.trial
        """
        raise "To be implemented"


class optimiser_momentum_strategy(trading_optimisation):
    """
    Optimizer for Momentum Strategy

    This class specializes in optimizing the Momentum strategy.

    """

    def __init__(self, config:dict) -> None:
        """
        Initialize Optimizer for Momentum Strategy

        This method initializes the optimizer for the Momentum strategy.

        :param config: Configuration parameters for optimization.
        :type config: dict
        """
        super().__init__(config)
        self.strategy = momentum_strategy(config) # initialize the momentum strategy

    def objective(self,trial:optuna.trial)->None:
        """
        Objective for Momentum Strategy Optimization

        This method defines the objective function for optimizing the Momentum strategy.

        :param trial: Optuna trial object.
        :type trial: optuna.trial
        :return: None
        :rtype: None
        """
        
        self.config["strategy"]["x"] = trial.suggest_int('x', 1, 34) # hyperparameters 
        self.config["strategy"]["y"] = trial.suggest_int('y', 35 ,120) # hyperparameters
        self.config["strategy"]["ratio_method"] = trial.suggest_categorical('ratio_method',["ma_ratio","ema_ratio"]) # hyperparameters
        self.strategy.change_config(self.config) # essential change config wil reinitialize the object
        simulation_df = self.strategy.compute_simulation() # compute the simulation
        result_portfolio =  simulation_df["Portfolio"].iloc[-1]
        rsi = (simulation_df[f"RSI_Portfolio/Buy_and_Hold"]).mean()
        result = result_portfolio*rsi # calculate a result
        config = copy.deepcopy(self.config)
        self.liste.append([config,result,result_portfolio,rsi,create_signature(self.config)])

        return result
    def save_df(self):
        df = pd.DataFrame(self.liste,columns=["config","result","result_portfolio","rsi","signature"])
        dataframe_path = self.strategy.time_eccefpsn + f"/{self.config['crypto_currency']}_{self.config['time']}_{self.config['strategy']['name']}.csv"
        if os.path.exists(dataframe_path):
            existing_df = pd.read_csv(dataframe_path)
            df = pd.concat([df,existing_df],ignore_index=True)
            del existing_df
        df = df.drop_duplicates(subset=['result'])
        df = df.drop_duplicates(subset=['signature'])
        df.to_csv(dataframe_path,index=False)
        self.liste = []


class optimiser_envelop_strategy(trading_optimisation):
    """
    Optimizer for Envelop Strategy

    This class specializes in optimizing the Momentum strategy.

    """

    def __init__(self, config:dict) -> None:
        """
        Initialize Optimizer for Momentum Strategy

        This method initializes the optimizer for the Momentum strategy.

        :param config: Configuration parameters for optimization.
        :type config: dict
        """
        super().__init__(config)
        self.strategy = envelop_strategy(config) # initialize the momentum strategy


    def objective(self,trial:optuna.trial)->None:
        """
        Objective for Momentum Strategy Optimization

        This method defines the objective function for optimizing the Momentum strategy.

        :param trial: Optuna trial object.
        :type trial: optuna.trial
        :return: None
        :rtype: None
        """
        
        self.config["strategy"]["x"] = trial.suggest_int('x', 1, 30) # hyperparameters
        self.config["strategy"]["method"] = trial.suggest_categorical('ratio_method',["ma","ema"]) # hyperparameters
        nb_envelope = trial.suggest_int(f"nb_enveloppe",2,5)
        list_envelope = []
        list_porportion = []
        for i in range(nb_envelope):
            if i ==0:
                list_envelope.append(trial.suggest_int(f"envelope_{i}",1,1000)*0.0001)
            else:
                list_envelope.append(trial.suggest_int(f"envelope_{i}",1,1000)*0.0001+list_envelope[-1])
            list_porportion.append(trial.suggest_int(f"proportion_envelope_{i}",0,100))

        envelope_array = np.round(np.array(list_envelope),3)
        porportion_array = np.array(list_porportion)
        porportion_array = np.round(porportion_array/np.sum(porportion_array),2)
            
        self.config["strategy"]["envelop"] = dict(zip(envelope_array.astype(str), porportion_array))
        self.strategy.change_config(self.config) # essential change config wil reinitialize the object
        simulation_df = self.strategy.compute_simulation() # compute the simulation
        
        col_1,col_2 = "Portfolio","Buy_and_Hold" # element for computing the reward
        simulation_df = evalutation(simulation_df,col_1,col_2) # evaluate the simulation
        # result = simulation_df["Portfolio"].iloc[-1]*(simulation_df[f"RSI_{col_1}/{col_2}"]/100).mean() # calculate a result
        return simulation_df["Portfolio"].iloc[-1]

            
if __name__=="__main__":
    absolute_path = os.path.abspath(__file__)[:os.path.abspath(__file__).find("MAD_BTC")+len("MAD_BTC")]

    df = optimiser_momentum_strategy({
    "crypto_currency": "OCEANUSDT",
    "time": "4h",
    "exchange": "binance",
    "begin_date":None,
    "end_date":None,
    "strategy": {
      "name": "momentum_strategy",
      "x": 5,
      "y": 57,
      "ratio_method": "ma_ratio"
    },
    "initial_capital": 1000,
    "broker": "binance"
  }).optimiser_multiple_currency()

        