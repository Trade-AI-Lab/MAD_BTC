import pandas as pd
from ta.momentum import RSIIndicator



def merge_strategies(list_strategies):

    # Concatenate the dataframes along the rows
    df_concat = pd.concat(list_strategies, ignore_index=True)

    # Sort the rows by date
    df_sorted = df_concat.sort_values(by='Time')

    # Fill the missing values with the previous ones
    return df_sorted.ffill().fillna(0)

def evalutation(df,column_A,column_B):
    df[f"{column_A}/{column_B}"] = df[column_A]/df[column_B]
    df[f"RSI_{column_A}/{column_B}"] = RSIIndicator(df[f"{column_A}/{column_B}"])
    return df

if __name__=="__main__":
    data_A = {'Time': ['2022-02-20', '2022-03-20'],
          'Portfolio_A': [1000, 1200]}
    df_strategy_A = pd.DataFrame(data_A)

    data_B = {'Time': ['2022-02-21', '2022-03-19'],
            'Portfolio_B': [1100, 1300]}
    df_strategy_B = pd.DataFrame(data_B)

    # Convert the Date columns to datetime objects
    df_strategy_A['Time'] = pd.to_datetime(df_strategy_A['Time'])
    df_strategy_B['Time'] = pd.to_datetime(df_strategy_B['Time'])
    print(merge_strategies([df_strategy_A,df_strategy_B]))

    """
    Merge two strategies and then calculate the ratio between 
    them and then does the RSI if RSI >50 means strat_a > strat_b overall
    """
    