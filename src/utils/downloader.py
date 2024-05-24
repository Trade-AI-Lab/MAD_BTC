import asyncio
import ccxt.async_support as ccxt
import pandas as pd
import datetime
import nest_asyncio
import toml

import os
nest_asyncio.apply()

with open('src/configs/download_config.toml', 'r') as file:
    CONFIG = toml.load(file)

async def _ohlcv(exchange, symbol, timeframe, limit, step_since, timedelta):
    result = await exchange.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit, since=step_since)
    result_df = pd.DataFrame(result, columns=["timestamp_open", "open", "high", "low", "close", "volume"])
    for col in ["open", "high", "low", "close", "volume"]:
        result_df[col] = pd.to_numeric(result_df[col])
    result_df["open_date"] = pd.to_datetime(result_df["timestamp_open"], unit="ms")
    #result_df["date_close"] = pd.to_datetime(result_df["timestamp_open"] + timedelta, unit="ms")

    return result_df


async def _download_symbol(exchange, symbol, timeframe='5m',
                           since=int(datetime.datetime(year=2020, month=1, day=1).timestamp() * 1E3),
                           until=int(datetime.datetime.now().timestamp() * 1E3), limit=1000, pause_every=10, pause=1):
    timedelta = int(pd.Timedelta(timeframe).to_timedelta64() / 1E6)
    tasks = []
    results = []
    for step_since in range(since, until, limit * timedelta):
        tasks.append(
            asyncio.create_task(_ohlcv(exchange, symbol, timeframe, limit, step_since, timedelta))
        )
        if len(tasks) >= pause_every:
            results.extend(await asyncio.gather(*tasks))
            await asyncio.sleep(pause)
            tasks = []
    if len(tasks) > 0:
        results.extend(await asyncio.gather(*tasks))
    final_df = pd.concat(results, ignore_index=True)
    final_df = final_df.loc[(since < final_df["timestamp_open"]) & (final_df["timestamp_open"] < until), :]
    final_df.drop_duplicates(inplace=True)
    return final_df


async def _download_symbols(exchange_name, symbols, dir, timeframes, **kwargs):
    exchange = getattr(ccxt, exchange_name)({'enableRateLimit': True})
    for symbol in symbols:
        for timeframe in timeframes:
            df = await _download_symbol(exchange=exchange, symbol=symbol, timeframe=timeframe, **kwargs)
            save_file = f"{dir}/{exchange_name}-{symbol.replace('/', '')}-{timeframe}.pkl"
            print(f"{symbol} for {timeframe} downloaded from {exchange_name} and stored at {save_file}")
            df.to_pickle(save_file)
    await exchange.close()


async def _download(exchange_names, symbols, timeframe, dir, since: datetime.datetime,
                    until: datetime.datetime = datetime.datetime.now()):
    tasks = []
    for exchange_name in exchange_names:
        limit = CONFIG[exchange_name]["limit"]
        pause_every = CONFIG[exchange_name]["pause_every"]
        pause = CONFIG[exchange_name]["pause"]
        tasks.append(
            _download_symbols(
                exchange_name=exchange_name, symbols=symbols, timeframes=timeframe, dir=dir,
                limit=limit, pause_every=pause_every, pause=pause,
                since=int(since.timestamp() * 1E3), until=int(until.timestamp() * 1E3)
            )
        )
    await asyncio.gather(*tasks)


def download(*args, **kwargs):
    loop = asyncio.get_event_loop()
    loop.run_until_complete(
        _download(*args, **kwargs)
    )

# Main function do modify
async def main(time_frame:str="5m"):
    absolute_path = os.path.abspath(__file__)[:os.path.abspath(__file__).find("MAD_BTC")+len("MAD_BTC")]
    # check if data folder exists
    dir = absolute_path + "/data"
    if not os.path.exists(dir):
        # Create the folder
        os.makedirs(dir)
        created_or_exists = "was created"
    else:
        created_or_exists = "already exists"
    # Get the name of the folder
    folder_name = os.path.basename(dir)
    print(f"The folder '{folder_name}' {created_or_exists}.")

    await _download(
        # TODO config files
        CONFIG["exchange"], # select your exchange
        #symbols=["ADAUSDT","ETHUSDT","BNBUSDT","BTCUSDT"], # select your currencies
        # TODO put this as parameter in config file
        symbols=CONFIG["symbols"],
        timeframe=time_frame, # the time frame you can change
        dir=dir, # directory
        since=datetime.datetime(year=2018, month=4, day=1), # starting date
    )


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    # TODO config file here 
    loop.run_until_complete(main(time_frame=CONFIG["time_frame"]))
    print("Done ")
    #loop.run_until_complete(main(time_frame="4h"))
    #loop.run_until_complete(main(time_frame="1h"))
    #loop.run_until_complete(main(time_frame="5m"))
    # asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    # asyncio.run(main())