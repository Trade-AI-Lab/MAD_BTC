import pandas as pd
from ta.momentum import RSIIndicator
import os, sys
absolute_path = os.path.abspath(__file__)[:os.path.abspath(__file__).find("MAD_BTC")+len("MAD_BTC")]
sys.path.append(absolute_path)
from src.utils.utils_df import open_dataframe
import matplotlib.pyplot as plt

def merge_strategies(list_strategies):

    # Concatenate the dataframes along the rows
    df_concat = pd.concat(list_strategies, ignore_index=True)

    # Sort the rows by date
    df_sorted = df_concat.sort_values(by='Time')

    # Fill the missing values with the previous ones
    return df_sorted.ffill().fillna(0).drop_duplicates(subset=['Time'])

def evalutation(df,column_A,column_B):
    """
    Merge two strategies and then calculate the ratio between 
    them and then does the RSI if RSI >50 means strat_a > strat_b overall
    """
    df[f"{column_A}/{column_B}"] = df[column_A]/df[column_B]
    df[f"RSI_{column_A}/{column_B}"] = RSIIndicator(df[f"{column_A}/{column_B}"]).rsi()
    return df

if __name__=="__main__":
    list_crypto_currency=["BTCUSDT","ETHUSDT","ADAUSDT","BNBUSDT"]
    list_time=["1h","4h","1d"]
    exchange = "binance"
    for crypto_currency in list_crypto_currency:
         for time in list_time:
            print("#################")
            print(f"Crypto_currency: {crypto_currency}\n")
            print(f"Time: {time}\n")
            print(f"exchange: {exchange}\n")
            exchange = "binance"
            PATH_FOLDER = absolute_path+rf"\experience\{crypto_currency}\{exchange}\{time}"
            liste = os.listdir(PATH_FOLDER)
            best = 0
            best_PATH = None
            col_1,col_2 = "Portfolio","Buy_and_Hold"
            for path in liste:
                PATH = os.path.join(PATH_FOLDER,path,'output_file.csv')
                df = open_dataframe(PATH,light=True)
                df = evalutation(df,col_1,col_2)
                result = (df[f"RSI_{col_1}/{col_2}"]/100).mean()
                if result>best:
                    best = result
                    best_PATH = PATH             
                    
            print(f"Final mean rsi: {best}\n")
            print(f"Link: {best_PATH}\n")
            plt.plot(df["Time"],df[f"RSI_{col_1}/{col_2}"])
            plt.title(f"RSI_{col_1}/{col_2}")
            plt.show()
            

    # Convert the Date columns to datetime objects
    #df_strategy_A['Time'] = pd.to_datetime(df_strategy_A['Time'])
    #df_strategy_A = df_strategy_A.rename(columns={'Portfolio': 'Portfolio_A'})
    #df_strategy_A = df_strategy_A.rename(columns={'Buy_and_Hold': 'Buy_and_Hold_A'})
    #df_strategy_B['Time'] = pd.to_datetime(df_strategy_B['Time'])
    #df_strategy_B = df_strategy_B.rename(columns={'Portfolio': 'Portfolio_B'})
    #df_strategy_B = df_strategy_B.rename(columns={'Buy_and_Hold': 'Buy_and_Hold_B'})
    
    #df = merge_strategies([df_strategy_A,df_strategy_B])
    #col_1,col_2 = "Portfolio_A","Portfolio_B"
    #df = evalutation(df,col_1,col_2)
    #import matplotlib.pyplot as plt
    #print((df[f"RSI_{col_1}/{col_2}"]/100).mean())
    #df.boxplot(column=f"RSI_{col_1}/{col_2}")
    #plt.show()
    #plt.plot(df["Time"],df[f"RSI_{col_1}/{col_2}"])
    #plt.show()
   
    