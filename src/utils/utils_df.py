import pandas as pd
from typing import Optional


def open_dataframe(PATH: str, light: bool = False) -> pd.DataFrame:
    """
    open_dataframe _summary_

    _extended_summary_

    :param PATH: _description_
    :type PATH: str
    :param light: _description_, defaults to False
    :type light: bool, optional
    :return: _description_
    :rtype: pd.DataFrame
    """
    extension = PATH.split(".")[-1]
    if extension == "zip" or extension == "csv":
        df = pd.read_csv(PATH)
    elif extension == "pkl":
        df = pd.read_pickle(PATH)
    else:
        raise "Error: unknown file type : ." + extension
    # Normalize columns name
    if not light:
        df.columns = df.columns.str.lower()
        if "close" not in df.columns:
            if "open_price" in df.columns:
                df["close"] = df["open_price"].shift(-1)
            else:
                raise "impossible to find an open price to create close price"

    if "open_date" not in df.columns:
        raise ValueError("The DataFrame does not contain a column named 'open_date'.")

    df["open_date"] = pd.to_datetime(df["open_date"])

    # Calculate time differences
    time_diffs = df["open_date"].diff(1)
    time_delta = df["open_date"].diff(1).iloc[-1]
    # Check if time differences are consistent
    is_continuous = (time_diffs[time_diffs != time_delta]).index

    if len(is_continuous) and light:
        print("Time is not continuous based on the 'open_date' column.")
    elif light:
        print("Time is  continuous based on the 'open_date' column.")
    return df


def y_build(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    y_build _summary_

    _extended_summary_

    :param df: _description_
    :type df: pd.DataFrame
    :param column: _description_
    :type column: str
    :return: _description_
    :rtype: pd.DataFrame
    """
    assert column in df.columns
    df[f"{column}_label"] = (df[column].diff(1) > 0).astype(int)
    return df


def filter_dataframe_by_date_range(
    dataframe: pd.DataFrame,
    begin_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    """
    filter_dataframe_by_date_range _summary_

    _extended_summary_

    :param dataframe: _description_
    :type dataframe: pd.DataFrame
    :param begin_date: _description_, defaults to None
    :type begin_date: Optional[str], optional
    :param end_date: _description_, defaults to None
    :type end_date: Optional[str], optional
    :return: _description_
    :rtype: _type_
    """
    if begin_date is not None and end_date is not None:
        df = dataframe[
            (dataframe["open_date"] >= begin_date)
            & (dataframe["open_date"] <= end_date)
        ]
    elif begin_date is not None:
        df = dataframe[(dataframe["open_date"] >= begin_date)]
    elif end_date is not None:
        df = dataframe[(dataframe["open_date"] <= end_date)]
    else:
        df = dataframe
    return df.reset_index(drop=True)


if __name__ == "__main__":
    PATH = r"./data/binance-BTCUSDT-5m.pkl"
    df = filter_dataframe_by_date_range(
        y_build(open_dataframe(PATH=PATH), column="close"),
        begin_date="2023-04-03",
        end_date="2023-05-06",
    )
    print(df.head())
