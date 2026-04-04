import asyncio
import ccxt.async_support as ccxt
import pandas as pd
import datetime
import nest_asyncio
import toml
import os

from tqdm.asyncio import tqdm_asyncio

nest_asyncio.apply()


# -----------------------------
# FETCH OHLCV
# -----------------------------
async def _ohlcv(exchange, symbol, timeframe, since, limit, retries=3):
    for attempt in range(retries):
        try:
            result = await exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=since,
                limit=limit,
            )

            if not result:
                return pd.DataFrame()

            df = pd.DataFrame(
                result,
                columns=["timestamp_open", "open", "high", "low", "close", "volume"],
            )

            df[["open", "high", "low", "close", "volume"]] = df[
                ["open", "high", "low", "close", "volume"]
            ].apply(pd.to_numeric)

            df["open_date"] = pd.to_datetime(df["timestamp_open"], unit="ms")

            return df

        except Exception as e:
            if attempt == retries - 1:
                print(f"❌ Failed {symbol} @ {since}: {e}")
                return pd.DataFrame()
            await asyncio.sleep(1)


# -----------------------------
# DOWNLOAD SINGLE SYMBOL
# -----------------------------
async def _download_symbol(
    exchange,
    symbol,
    timeframe,
    since,
    until,
    limit,
    semaphore,
):
    delta = int(pd.Timedelta(timeframe).total_seconds() * 1000)
    step = limit * delta

    tasks = []

    for step_since in range(since, until, step):

        async def bound_fetch(s=step_since):
            async with semaphore:
                return await _ohlcv(exchange, symbol, timeframe, s, limit)

        tasks.append(bound_fetch())

    results = await tqdm_asyncio.gather(
        *tasks,
        desc=f"{symbol}-{timeframe}",
        leave=False,
    )

    # Filter + concat safely
    dfs = [r for r in results if not r.empty]

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)

    df = df[
        (df["timestamp_open"] > since) & (df["timestamp_open"] < until)
    ].drop_duplicates()

    return df


# -----------------------------
# DOWNLOAD MULTIPLE SYMBOLS
# -----------------------------
async def _download_symbols(exchange_name, symbols, folder, timeframes, **kwargs):
    exchange = getattr(ccxt, exchange_name)({"enableRateLimit": True})

    semaphore = asyncio.Semaphore(5)  # 🔥 tune this (3–10)

    for symbol in symbols:
        for timeframe in timeframes:
            print(f"⬇️ Downloading {symbol} [{timeframe}] from {exchange_name}")

            df = await _download_symbol(
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                semaphore=semaphore,
                **kwargs,
            )

            if df.empty:
                print(f"⚠️ No data for {symbol} {timeframe}")
                continue

            save_file = (
                f"{folder}/{exchange_name}-{symbol.replace('/', '')}-{timeframe}.pkl"
            )

            df.to_pickle(save_file)

            print(f"✅ Saved -> {save_file}")

    await exchange.close()


# -----------------------------
# MAIN DOWNLOAD
# -----------------------------
async def _download(config):
    since = int(datetime.datetime.fromisoformat(config["since"]).timestamp() * 1e3)
    until = int(datetime.datetime.fromisoformat(config["until"]).timestamp() * 1e3)

    tasks = []

    for exchange_name in config["exchange"]:
        tasks.append(
            _download_symbols(
                exchange_name=exchange_name,
                symbols=config["symbols"],
                timeframes=config["time_frame"],
                folder=config["folder"],
                since=since,
                until=until,
                limit=config[exchange_name]["limit"],
            )
        )

    await asyncio.gather(*tasks)


# -----------------------------
# ENTRY POINT
# -----------------------------
async def downloader(config: dict):
    base_path = os.path.abspath(__file__)
    root = base_path[: base_path.find("MAD_BTC") + len("MAD_BTC")]

    folder = os.path.join(root, config["folder"])
    config["folder"] = folder

    os.makedirs(folder, exist_ok=True)

    print(f"📁 Data folder ready: {folder}")

    await _download(config)


# -----------------------------
# RUN SCRIPT
# -----------------------------
if __name__ == "__main__":
    with open("src/configs/download_config.toml", "r") as f:
        config = toml.load(f)

    asyncio.run(downloader(config))

    print("🎉 Done")
