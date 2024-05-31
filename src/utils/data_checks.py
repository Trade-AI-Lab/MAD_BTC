import pickle
import pandas as pd
import os
import itertools
from collections import defaultdict
import sys

sys.path.append("./")
from src.utils.custom_exceptions import DateMismatch


def get_open_date_values(file_path):
    try:
        with open(file_path, "rb") as file:
            df = pickle.load(file)
            if isinstance(df, pd.DataFrame) and "open_date" in df.columns:
                first_open_date = df["open_date"].iloc[0]
                last_open_date = df["open_date"].iloc[-1]
                return first_open_date, last_open_date
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
    return None


def check_open_dates_consistency(folder, files):
    # TODO handle the error for mismatch data (adapt the dataset if possible)
    first_date, last_date = None, None
    bad_dataset = []
    for file in files:
        dates = get_open_date_values(os.path.join(folder, file))
        if dates:
            if first_date is None and last_date is None:
                first_date, last_date = dates
            else:
                if dates[0] != first_date or dates[1] != last_date:
                    bad_dataset.append(file)
    # if not bad_dataset:
    #     return
    # else:
    #     raise DateMismatch(bad_dataset)


def check_data_availability_in_folder(folder, brokers, crypto_currencies, times):
    # Convert lists to sets for faster lookup
    set_brokers = set(brokers)
    set_crypto_currencies = set(crypto_currencies)
    set_times = set(times)

    # Initialize a set to store found combinations
    found_combinations = set()

    # Process each file in the folder
    for file_name in os.listdir(folder):
        if file_name.endswith(".pkl"):
            # Extract the parts from the filename
            title_parts = os.path.splitext(file_name)[0].split("-")
            if len(title_parts) == 3:
                broker = title_parts[0]
                crypto = title_parts[1]
                time = title_parts[2]

                # Check if the extracted parts are in the target sets
                if (
                    broker in set_brokers
                    and crypto in set_crypto_currencies
                    and time in set_times
                ):
                    found_combinations.add((broker, crypto, time))

    # Generate all possible combinations using Cartesian product
    all_combinations = set(itertools.product(brokers, crypto_currencies, times))

    # Determine missing combinations
    missing_combinations = all_combinations - found_combinations

    # Print missing combinations
    for broker, crypto, time in missing_combinations:
        print(
            f"Data is missing for broker {broker}, cryptocurrency {crypto}, and time {time}."
        )
    data_to_download = sort_download_data(missing_combinations)

    available_data = {
        f"{broker}-{crypto}-{time}.pkl" for broker, crypto, time in found_combinations
    }
    # check if the dates corresponds between the datasets:
    # TODO adapt this one and change to a try except
    check_open_dates_consistency(folder, available_data)
    return data_to_download


def sort_download_data(data):
    # Initialize sets to avoid duplicates
    currencies_set = set()
    periods_set = set()
    broker_set = set()

    # Populate the sets with unique currencies and periods
    # Assume there only is one broker here.
    for broker, crypto, period in data:
        currencies_set.add(crypto)
        periods_set.add(period)
        broker_set.add(broker)

    # Convert sets to sorted lists
    currencies_list = sorted(currencies_set)
    periods_list = sorted(periods_set)
    broker_list = sorted(broker_set)

    return currencies_list, periods_list, broker_list


if __name__ == "__main__":
    # Example usage
    list_brokers = ["binance"]
    list_crypto_currency = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "BNBUSDT"]
    list_time = ["1h", "4h", "1d"]
    folder = "data"

    data_to_download = check_data_availability_in_folder(
        folder,
        brokers=list_brokers,
        crypto_currencies=list_crypto_currency,
        times=list_time,
    )
    print(data_to_download)
