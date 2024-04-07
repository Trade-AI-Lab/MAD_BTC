import os
import sys
import pandas as pd
absolute_path = os.path.abspath(__file__)[:os.path.abspath(__file__).find("MAD_BTC")+len("MAD_BTC")]
sys.path.append(absolute_path)
from src.utils.utils_df import open_dataframe, filter_dataframe_by_date_range
from src.strategy.momentum import indicator_momentum
from src.strategy.envelop import indicator_envelop
from src.strategy_utils.renderer_strategy import renderer,save
from src.strategy_utils.save_experiences_strategy import create_save_one_experience,create_save_experiences
from src.utils.utils_signature import create_signature

class trading_strategy:
    """
    Trading Strategy

    This class represents a generic trading strategy.
    """
    def __init__(self,config:dict) -> None:
        """
        Initialize Trading Strategy

        This method initializes the trading strategy.

        :param config: Configuration parameters for the strategy.
        :type config: dict
        """
        self.config = config
        self.PATH = absolute_path + rf"/data/{config['exchange']}-{config['crypto_currency']}-{config['time']}.pkl"
        self.parent_df = open_dataframe(self.PATH)
        self.child_df = None
        self.df_filtered = None
        self.strategy_folder_path = None
        self.time_eccefpsn = create_save_experiences(self.config)
        self.evolution_df = None
        self.signature_strategy = create_signature(self.config["strategy"])
        self.begin_date = self.config["begin_date"] if "begin_date" in self.config.keys() else None
        self.end_date = self.config["end_date"] if "end_date" in self.config.keys() else None

    
    def loop_simulation(self)->None:
        """
        loop_simulation loop_simulation represents a trading strategy implementation that simulates buying and selling of a cryptocurrency.

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


        self.child_df.set_index('open_date', inplace=True)
        self.df_filtered.set_index('open_date', inplace=True)

        for index, row in self.df_filtered.iterrows():
            row = self.df_filtered.loc[index]
            if row['buy_signal'] and position != 1 and portfolio > 0:
                position = 1
                list_portfolio.append(portfolio)
                list_position.append(position)
                crypto = portfolio / (row['open'] * (1 + fee_percentage))
                portfolio = 0

                list_buy_hold.append(row['buy_and_hold'])
                list_time.append(index)

            elif row['sell_signal'] and position != -1 and crypto > 0:
                position = -1
                portfolio = crypto * row['open'] * (1 - fee_percentage)
                crypto = 0

                list_portfolio.append(portfolio)
                list_position.append(position)
                list_buy_hold.append(row['buy_and_hold'])
                list_time.append(index)

            elif crypto == 0 and portfolio == 0:
                break 
        


        if crypto > 0:
            # Calculate the final portfolio value based on remaining position in crypto
            final_portfolio_value = crypto * self.child_df.iloc[-1]['open'] * (1 - fee_percentage)
            list_portfolio.append(final_portfolio_value)
            list_position.append(position)  # Assuming you maintain the same position
            list_buy_hold.append(self.child_df['buy_and_hold'].iloc[-1])
            list_time.append(self.child_df.index[-1])

        elif crypto == 0:
            list_portfolio.append(list_portfolio[-1])
            list_position.append(list_position[-1])  # Assuming you maintain the same position
            list_buy_hold.append(self.child_df['buy_and_hold'].iloc[-1])
            list_time.append(self.child_df.index[-1])

        
        self.evolution_df = pd.DataFrame({
        "Time":list_time,
        'Portfolio': list_portfolio,
        'Buy_and_Hold': list_buy_hold,
        "Position":list_position})
        self.strategy_folder_path = create_save_one_experience(self.config,self.time_eccefpsn)
        save("No strategy",self.strategy_folder_path,self.evolution_df) if self.title_image is None else save(self.title_image,self.strategy_folder_path,self.evolution_df)
    
    def compute_alpha_dates(self, df):
        df["time_diffs"] = df["open_date"].diff(-1)
        time_delta = df["open_date"].diff(1).iloc[-1]
        return (df["time_diffs"] == -time_delta)

    
    def renderer(self) -> None:
        """
        Renderer

        This method renders the evolution of the strategy.

        :return: None
        :rtype: None
        """
        renderer(self.evolution_df,verbose=True)

    
    def compute_simulation(self)->pd.DataFrame:
        """
        compute_simulation _summary_

        _extended_summary_

        :return: _description_
        :rtype: pd.DataFrame
        """
        self.compute_needed_features()
        self.strategy_rules()
        self.loop_simulation()
        return self.evolution_df
    
    def compute_needed_features(self)->None:
        """
        Compute Needed Features for Trading Strategy.

        This method calculates the necessary features for the Trading Strategy...

        """
        self.child_df = filter_dataframe_by_date_range(self.parent_df.copy(deep=True),self.begin_date,self.end_date)
        self.child_df['buy_and_hold'] = self.child_df['open'] / self.child_df['open'].iloc[0] * self.config["initial_capital"]
    
    def strategy_rules(self):
        """
        Define Strategy Rules.

        This method defines the rules for the momentum strategy, including buy and sell signals.

        """
        raise "To be implemented"
    
class momentum_strategy(trading_strategy):
    """
    Momentum Trading Strategy

    This class represents a momentum trading strategy based on the Moving Average Distance (MAD) concept.

    :param trading_strategy: The base class for trading strategies.
    :type trading_strategy: trading_strategy
    """
    def __init__(self,config) -> None:
        """
        Initialize the Momentum Trading Strategy.

        :param config: Configuration parameters for the strategy.
        :type config: dict
        """
        super().__init__(config)
        
    def compute_needed_features(self):
        """
        Compute Needed Features for Momentum Strategy.

        This method calculates the necessary features for the momentum strategy, including the MAD ratio.

        """
        super().compute_needed_features()
        self.strategy_name = f'{self.config["strategy"]["ratio_method"]}_{self.config["strategy"]["x"]}_{self.config["strategy"]["y"]}'
        self.title_image = f"{self.config['exchange']}-{self.config['crypto_currency']}-{self.config['time']}_portfolio_vs_buy_hold_{self.strategy_name}_{self.config['initial_capital']}$"
        self.child_df[self.strategy_name] = getattr(indicator_momentum,self.config["strategy"]["ratio_method"])(self.child_df, self.config["strategy"]["x"],self.config["strategy"]["y"])
        self.child_df = self.child_df.dropna().reset_index(drop=True)
    
    def strategy_rules(self):
        """
        Define Strategy Rules.

        This method defines the rules for the momentum strategy, including buy and sell signals.

        """
        self.compute_needed_features() if self.child_df is None else None
        self.child_df['buy_signal'] = (self.child_df[self.strategy_name] > 1) & (self.child_df[self.strategy_name].shift(1) <= 1)
        self.child_df['sell_signal'] = (self.child_df[self.strategy_name] < 1) & (self.child_df[self.strategy_name].shift(1) >= 1)
        self.df_filtered = self.child_df[(self.child_df['buy_signal']>0) | (self.child_df['sell_signal']>0)].reset_index(drop=True)

    def change_config(self,config):
        """
        Change Configuration.

        This method allows the user to change the configuration, but the object needs to be reinitialized.

        :param config: New configuration for the strategy.
        :type config: dict
        """
        if self.PATH == absolute_path + rf"/data/{config['exchange']}-{config['crypto_currency']}-{config['time']}.pkl":
            if self.signature_strategy != create_signature(self.config["strategy"]):
                self.child_df = None 
        else:
            super(momentum_strategy, self).__init__(config)

    
class envelop_strategy(trading_strategy):
    """
    Envelop Trading Strategy

    This class represents a envelop trading strategy.

    :param trading_strategy: The base class for trading strategies.
    :type trading_strategy: trading_strategy
    """
    def __init__(self,config) -> None:
        """
        Initialize the Momentum Trading Strategy.

        :param config: Configuration parameters for the strategy.
        :type config: dict
        """
        super().__init__(config)
        

    def compute_needed_features(self):
        """
        Compute Needed Features for Momentum Strategy.

        This method calculates the necessary features for the momentum strategy, including the MAD ratio.

        """
        super().compute_needed_features()
        envelop = self.config["strategy"]["envelop"]
        self.strategy_name = f'{self.config["strategy"]["method"]}_{self.envelope_from_list_to_str(envelop)}'
        self.title_image = f"{self.config['exchange']}-{self.config['crypto_currency']}-{self.config['time']}_portfolio_vs_buy_hold_{self.strategy_name}_{self.config['initial_capital']}$"
        self.child_df[self.strategy_name] = getattr(indicator_envelop,self.config["strategy"]["method"])(self.child_df, self.config["strategy"]["x"]) # refers to the MA or EMA
        self.child_df = self.child_df.dropna().reset_index(drop=True)
        self.child_df["alpha_date"] = self.compute_alpha_dates(self.child_df)
        self.list_col_buy_signal = []
        self.list_col_sell_signal = []

        for percent_envelop in self.config["strategy"]["envelop"]:
            
            percent_portfolio = self.config["strategy"]["envelop"][percent_envelop]
            percent_envelop = float(percent_envelop)
            # Buy
            self.child_df[f"envelop_up_{percent_envelop}%"] = self.child_df[self.strategy_name]*(1+percent_envelop) # E and/or MA(1+x%)
            buy_signal = f"buy_signal_{percent_envelop}%"
            self.child_df[buy_signal] = (self.child_df["high"]>=self.child_df[f"envelop_up_{percent_envelop}%"]).astype(int)*percent_portfolio
            self.list_col_buy_signal.append(buy_signal)

            # Sell
            self.child_df[f"envelop_down_{percent_envelop}%"] = self.child_df[self.strategy_name]*(1-percent_envelop) # E and/or MA(1-x%)
            sell_signal = f"sell_signal_{percent_envelop}%"
            self.child_df[sell_signal] = (self.child_df[f"envelop_down_{percent_envelop}%"]<=self.child_df["low"]).astype(int)*percent_portfolio
            self.list_col_sell_signal.append(sell_signal)

        
        
        
    
    def strategy_rules(self):
        """
        Define Strategy Rules.

        This method defines the rules for the momentum strategy, including buy and sell signals.

        """
        self.compute_needed_features() if self.child_df is None else None
        self.child_df["buy_signal"] =  self.child_df[self.list_col_buy_signal].sum(axis=1)
        self.child_df['sell_signal'] = self.child_df[self.list_col_sell_signal].sum(axis=1)
        self.df_filtered = self.child_df[(self.child_df['buy_signal']>0) | (self.child_df['sell_signal']>0)].reset_index(drop=True)

    def change_config(self,config):
        """
        Change Configuration.

        This method allows the user to change the configuration, but the object needs to be reinitialized.

        :param config: New configuration for the strategy.
        :type config: dict
        """
        if self.PATH == absolute_path + rf"/data/{config['exchange']}-{config['crypto_currency']}-{config['time']}.pkl":
            if self.signature_strategy != create_signature(self.config["strategy"]):
                self.child_df = None 
        else:
            super(envelop_strategy, self).__init__(config)

    @staticmethod
    def envelope_from_list_to_str(dictionnary:dict):
        liste = [key for key in dictionnary]
        return "_layer_".join(map(str, liste)) + "_layer"
    

    def loop_simulation(self)->None:
        """
        loop_simulation loop_simulation represents a trading strategy implementation that simulates buying and selling of a cryptocurrency.

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
        fee_maker = 0.2/100 # 0.2%
        crypto,position = 0,0
        list_portfolio,list_buy_hold,list_time,list_position = [],[],[],[]
        last_time = self.child_df["open_date"].iloc[-1]
        self.delta_t = self.child_df["open_date"].diff(1).iloc[-1]
        jumping_time = self.df_filtered["open_date"].iloc[0]
        self.df_filtered.set_index('open_date', inplace=True)
        self.child_df.set_index('open_date',inplace=True)
        end_simulation = False
        jumping_simulation = False
        for index, row in self.df_filtered.iterrows():
            row = self.df_filtered.loc[index]
            if index>jumping_time and end_simulation!=True: # if the jumping trade is higher than index you can trade
                k = 0
                time = self.delta_t*k + index
                if row['buy_signal']:
                    # Begin the trade
                    position = 1
                    for percent_envelop in self.config["strategy"]["envelop"]:
                        percent_portfolio = self.config["strategy"]["envelop"][percent_envelop]
                        crypto +=   portfolio*percent_portfolio / (row[f"envelop_up_{percent_envelop}%"]* (1 + fee_percentage)) # (1+fee_maker)>1 row[f"envelop_up_{percent_envelop}%"]*(1+fee_maker)>row[f"envelop_up_{percent_envelop}%"] means price higher implies crypto more expensive
                        portfolio-=portfolio*percent_portfolio
                    while self.child_df[self.strategy_name].loc[time]<self.child_df["low"].loc[time] and end_simulation!=True and jumping_simulation!=True and self.child_df["alpha_date"][time]:
                        k+=1
                        time = self.delta_t*k + index
                        if time == last_time :
                            end_simulation = True
                        elif self.child_df["alpha_date"][time]!=True:
                            jumping_simulation = True
                    # Trade ended
                    if end_simulation:
                        portfolio += crypto*self.child_df["open"].loc[last_time]*(1-fee_maker)
                        jumping_time = time

                    elif jumping_simulation:
                        portfolio += crypto*self.child_df["open"].loc[last_time]*(1-fee_maker)
                        jumping_time = time - self.child_df["time_diffs"].loc[time]
                        jumping_simulation = False

                    else:
                        portfolio += crypto*self.child_df[self.strategy_name].loc[time]*(1-fee_maker)
                        jumping_time = time
                        
                elif row['sell_signal']:
                    # Begin the trade
                    position = -1
                    for percent_envelop in self.config["strategy"]["envelop"]:
                        percent_portfolio = self.config["strategy"]["envelop"][percent_envelop]
                        crypto -=   portfolio*percent_portfolio / (row[f"envelop_down_{percent_envelop}%"]* (1 - fee_percentage) )
                        portfolio-=portfolio*percent_portfolio

                    
                    while self.child_df[self.strategy_name].loc[time]<self.child_df["low"].loc[time] and end_simulation!=True and jumping_simulation!=True and self.child_df["alpha_date"][time]:
                            k+=1
                            time = self.delta_t*k + index
                            if time == last_time :
                                end_simulation = True
                            elif self.child_df["alpha_date"][time]!=True:
                                jumping_simulation = True
                            
                    if end_simulation:
                        rebuy_price_crypto = portfolio*percent_portfolio/(self.child_df["open"].loc[time]* (1 + fee_maker)) 
                        portfolio += rebuy_price_crypto-crypto
                        jumping_time = time
                    
                    elif jumping_simulation:
                        rebuy_price_crypto = portfolio*percent_portfolio/(self.child_df["open"].loc[time]* (1 + fee_maker)) 
                        portfolio += rebuy_price_crypto-crypto
                        jumping_time = time - self.child_df["time_diffs"].loc[time]
                        jumping_simulation = False
                    else:
                        # Trade ended
                        rebuy_price_crypto = portfolio*percent_portfolio/(self.child_df[self.strategy_name].loc[time]* (1 + fee_maker)) 
                        portfolio += rebuy_price_crypto-crypto
                        jumping_time = time
                    
                list_portfolio.append(portfolio)
                list_position.append(position)
                list_buy_hold.append(self.child_df["buy_and_hold"].loc[time])
                list_time.append(time)

                crypto = 0
                position = 0
            elif len(list_portfolio)>0 and list_portfolio[-1]<=1:
                break

    
    
        
        self.evolution_df = pd.DataFrame({
        "Time":list_time,
        'Portfolio': list_portfolio,
        'Buy_and_Hold': list_buy_hold,
        "Position":list_position})
        self.strategy_folder_path = create_save_one_experience(self.config,self.time_eccefpsn)
        save("No strategy",self.strategy_folder_path,self.evolution_df) if self.title_image is None else save(self.title_image,self.strategy_folder_path,self.evolution_df)




        
    