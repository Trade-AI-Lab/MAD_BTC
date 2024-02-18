import os
import sys
import pandas as pd
absolute_path = os.path.abspath(__file__)[:os.path.abspath(__file__).find("MAD_BTC")+len("MAD_BTC")]
sys.path.append(absolute_path)
from src.utils.utils_df import open_dataframe
from src.strategy.momentum import indicator_momentum
from src.strategy_utils.renderer_strategy import renderer,save
from src.strategy_utils.save_experiences_strategy import create_save_one_experience,create_save_experiences

class trading_strategy:
    def __init__(self,config) -> None:
        self.config = config
        self.PATH = absolute_path + rf"/data/{config['exchange']}-{config['crypto_currency']}-{config['time']}.pkl"
        self.parent_df = open_dataframe(self.PATH)
        self.child_df = None
        self.df_filtered = None
        self.strategy_folder_path = None
        self.time_eccefp = create_save_experiences(self.config)
        self.title_image = f"{self.config['exchange']}-{self.config['crypto_currency']}-{self.config['time']}_portfolio_vs_buy_hold_{self.config['ratio_method']}_{self.config['x']}_{self.config['y']}_{self.config['initial_capital']}$"

    def compute_needed_features(self)->None:
        self.child_df = self.parent_df.copy(deep=True)
        self.child_df['buy_and_hold'] = self.child_df['open'] / self.child_df['open'][0] * self.config["initial_capital"]
        return None
    
    def strategy_rules(self)->pd.Dataframe:
        raise "To be implemented"
    
    def loop_simulation(self)->None:

        portfolio = self.config["initial_capital"]
        fee_percentage = 0.1 / 100  # 0.1%
        crypto,position = 0,0
        list_portfolio,list_buy_hold,list_time = [],[],[]

        for i in range(len(self.df_filtered)):

            if self.df_filtered.iloc[i]['buy_signal'] and position != 1 and portfolio > 0:
                position = 1
                crypto = portfolio / (self.df_filtered.iloc[i]['open']*(1 + fee_percentage))
                portfolio = 0

            elif self.df_filtered.iloc[i]['sell_signal'] and position != -1 and crypto > 0:
                position = -1
                portfolio = crypto * self.df_filtered.iloc[i]['open'] * (1-fee_percentage)
                crypto = 0

                
                list_portfolio.append(portfolio)
                list_buy_hold.append(self.df_filtered['buy_and_hold'].iloc[i])
                list_time.append(self.df_filtered["open_date"].iloc[i])

            elif crypto == 0 and portfolio == 0:
                break 
        
        list_portfolio.append(list_portfolio[-1])
        list_buy_hold.append(self.child_df['open'].iloc[-1] / self.child_df['open'][0] * self.config["initial_capital"])
        list_time.append(self.child_df["open_date"].iloc[-1])
        
        evolution_df = pd.DataFrame({
        "Time":list_time,
        'Portfolio': list_portfolio,
        'Buy_and_Hold': list_buy_hold})
        self.strategy_folder_path = self.create_save_one_experience(self.config,self.time_eccefp)
        save(self.title,self.strategy_folder_path,evolution_df)
        return evolution_df
    
    def renderer(self)->None:
        raise "To be implemented"
    
class momentum_strategy(trading_strategy):
    def __init__(self,config) -> None:
        super().__init__(config)

    def compute_needed_features(self)->None:
        super().compute_needed_features()

        self.child_df[f'{self.config["ratio_method"]}_{self.config["x"]}_{self.config["y"]}'] = getattr(indicator_momentum,self.config["ratio_method"])(self.child_df, self.config["x"],self.config["y"])
        self.child_df = self.child_df.dropna().reset_index(drop=True)
    
    def strategy_rules(self):
        self.compute_needed_features() if self.child_df is None else None
        self.child_df['buy_signal'] = (self.child_df[f'{self.config["ratio_method"]}_{self.config["x"]}_{self.config["y"]}'] > 1) & (self.child_df[f'{self.config["ratio_method"]}_{self.config["x"]}_{self.config["y"]}'].shift(1) <= 1)
        self.child_df['sell_signal'] = (self.child_df[f'{self.config["ratio_method"]}_{self.config["x"]}_{self.config["y"]}'] < 1) & (self.child_df[f'{self.config["ratio_method"]}_{self.config["x"]}_{self.config["y"]}'].shift(1) >= 1)
        self.df_filtered = self.child_df[self.child_df['buy_signal'] | self.child_df['sell_signal']].reset_index(drop=True)
    
    def renderer(self) -> None:
        
        renderer(title_image,self.strategy_folder_path,self.evolution_df)