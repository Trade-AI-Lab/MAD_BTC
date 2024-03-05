import os
import sys
import pandas as pd
absolute_path = os.path.abspath(__file__)[:os.path.abspath(__file__).find("MAD_BTC")+len("MAD_BTC")]
sys.path.append(absolute_path)
from src.utils.utils_df import open_dataframe
from src.strategy.momentum import indicator_momentum
from src.strategy_utils.renderer_strategy import renderer,save
from src.strategy_utils.save_experiences_strategy import create_save_one_experience,create_save_experiences
from src.utils.utils_signature import create_signature

class trading_strategy:
    """trading strategy is the core object to create trading strategies

    """
    def __init__(self,config) -> None:
        self.config = config
        self.PATH = absolute_path + rf"/data/{config['exchange']}-{config['crypto_currency']}-{config['time']}.pkl"
        self.parent_df = open_dataframe(self.PATH)
        self.child_df = None
        self.df_filtered = None
        self.strategy_folder_path = None
        self.time_eccefpsn = create_save_experiences(self.config)
        self.evolution_df = None
        self.signature_strategy = create_signature(self.config["strategy"])

    
    def loop_simulation(self)->None:
        """
        loop_simulation represents a trading strategy implementation that simulates buying and selling of a cryptocurrency.
        It initializes a portfolio with a specified initial capital, defines a fee percentage for transactions,
        and sets initial values for cryptocurrency holdings and position.
        The strategy iterates through a filtered DataFrame of market data, executing buy and sell signals based on certain conditions.
        It maintains track of the portfolio, position, and other relevant information over time.
        The resulting evolution of the portfolio and associated metrics are stored in a Pandas DataFrame.
        The code also creates a folder to save the strategy's results and visualizations.
        The final results are saved, and the code appears to be part of a larger trading strategy analysis or backtesting framework.
        """

        portfolio = self.config["initial_capital"]
        fee_percentage = 0.1 / 100  # 0.1%
        crypto,position = 0,0
        list_portfolio,list_buy_hold,list_time,list_position = [],[],[],[]

        for i in range(len(self.df_filtered)):

            if self.df_filtered.iloc[i]['buy_signal'] and position != 1 and portfolio > 0:
                position = 1
                list_portfolio.append(portfolio)
                list_position.append(position)
                crypto = portfolio / (self.df_filtered.iloc[i]['open']*(1 + fee_percentage))
                portfolio = 0

                
                list_buy_hold.append(self.df_filtered['buy_and_hold'].iloc[i])
                list_time.append(self.df_filtered["open_date"].iloc[i])

            elif self.df_filtered.iloc[i]['sell_signal'] and position != -1 and crypto > 0:
                position = -1
                portfolio = crypto * self.df_filtered.iloc[i]['open'] * (1-fee_percentage)
                crypto = 0

                
                list_portfolio.append(portfolio)
                list_position.append(position)
                list_buy_hold.append(self.df_filtered['buy_and_hold'].iloc[i])
                list_time.append(self.df_filtered["open_date"].iloc[i])

            elif crypto == 0 and portfolio == 0:
                break 
        
        list_portfolio.append(list_portfolio[-1])
        list_position.append(list_position[-1])
        list_buy_hold.append(self.child_df['buy_and_hold'].iloc[-1])
        list_time.append(self.child_df["open_date"].iloc[-1])
        
        self.evolution_df = pd.DataFrame({
        "Time":list_time,
        'Portfolio': list_portfolio,
        'Buy_and_Hold': list_buy_hold,
        "Position":list_position})
        self.strategy_folder_path = create_save_one_experience(self.config,self.time_eccefpsn)
        save("No strategy",self.strategy_folder_path,self.evolution_df) if self.title_image is None else save(self.title_image,self.strategy_folder_path,self.evolution_df)
    
    def renderer(self) -> None:
        """renderer is the function for plotting, visualizing
        """
        renderer(self.evolution_df,verbose=True)

    
    

    def compute_simulation(self)->pd.DataFrame:
        self.compute_needed_features()
        self.strategy_rules()
        self.loop_simulation()
        return self.evolution_df
    
    def compute_needed_features(self)->None:
        """Compute needed features, when you have a dataframe with OHLC 
        you need to compute needed features such as RSI or EMA for your trading strategies
        """
        self.child_df = self.parent_df.copy(deep=True)
        self.child_df['buy_and_hold'] = self.child_df['open'] / self.child_df['open'].iloc[0] * self.config["initial_capital"]
    
    def strategy_rules(self):
        """Compute rules for the trading strategy
        """
        raise "To be implemented"
    
class momentum_strategy(trading_strategy):
    def __init__(self,config) -> None:
        super().__init__(config)
        

    def compute_needed_features(self):
        super().compute_needed_features()
        self.strategy_name = f'{self.config["strategy"]["ratio_method"]}_{self.config["strategy"]["x"]}_{self.config["strategy"]["y"]}'
        self.title_image = f"{self.config['exchange']}-{self.config['crypto_currency']}-{self.config['time']}_portfolio_vs_buy_hold_{self.strategy_name}_{self.config['initial_capital']}$"
        self.child_df[self.strategy_name] = getattr(indicator_momentum,self.config["strategy"]["ratio_method"])(self.child_df, self.config["strategy"]["x"],self.config["strategy"]["y"])
        self.child_df = self.child_df.dropna().reset_index(drop=True)
    
    def strategy_rules(self):
        self.compute_needed_features() if self.child_df is None else None
        self.child_df['buy_signal'] = (self.child_df[self.strategy_name] > 1) & (self.child_df[self.strategy_name].shift(1) <= 1)
        self.child_df['sell_signal'] = (self.child_df[self.strategy_name] < 1) & (self.child_df[self.strategy_name].shift(1) >= 1)
        self.df_filtered = self.child_df[self.child_df['buy_signal'] | self.child_df['sell_signal']].reset_index(drop=True)

    def change_config(self,config):
        """change_config allows to the user to change the configuration but it is needed to reinitialize the object

        Args:
            config (dict): configuration containing the main important elements for the trading strategy
        """
        if self.PATH == absolute_path + rf"/data/{config['exchange']}-{config['crypto_currency']}-{config['time']}.pkl":
            if self.signature_strategy != create_signature(self.config["strategy"]):
                self.child_df = None 
        else:
            super(momentum_strategy, self).__init__(config)

    


class donchian_strategy(trading_strategy):
    def __init__(self,config) -> None:
        super().__init__(config)

    def compute_needed_features(self):
        super().compute_needed_features()
        self.strategy_name = f'{self.config["strategy"]["window"]}'
        self.title_image = f"{self.config['exchange']}-{self.config['crypto_currency']}-{self.config['time']}_portfolio_vs_buy_hold_{self.strategy_name}_{self.config['initial_capital']}$"
        donchian = indicator_momentum.donchian(self.child_df, self.config["strategy"]["window"])
        self.child_df[f"dcu_{self.strategy_name}"] = donchian[0]
        self.child_df[f"dcm_{self.strategy_name}"] = donchian[1]
        self.child_df[f"dcl_{self.strategy_name}"] = donchian[2]
        self.child_df = self.child_df.dropna().reset_index(drop=True)
    
    def strategy_rules(self):
        self.compute_needed_features() if self.child_df is None else None
        self.child_df['buy_signal'] = self.child_df[f"dcu_{self.strategy_name}"]>=self.child_df["high"]
        self.child_df['sell_signal'] = self.child_df[f"dcl_{self.strategy_name}"]<=self.child_df["low"]
        self.df_filtered = self.child_df[self.child_df['buy_signal'] | self.child_df['sell_signal']].reset_index(drop=True)

    def change_config(self,config):
        """change_config allows to the user to change the configuration but it is needed to reinitialize the object

        Args:
            config (dict): configuration containing the main important elements for the trading strategy
        """
        if self.PATH == absolute_path + rf"/data/{config['exchange']}-{config['crypto_currency']}-{config['time']}.pkl":
            if self.signature_strategy != create_signature(self.config["strategy"]):
                self.child_df = None 
        else:
            super(donchian_strategy, self).__init__(config)
        
    