import toml

config = {
    "bitget":
        {
            "limit":100,
            "pause_every":1,
            "pause":3
        },
    "bitfinex2": {
        "limit": 10_000,
        "pause_every": 1,
        "pause": 3,  # seconds
    },
    "binance": {
        "limit": 1_000,
        "pause_every": 1,
        "pause": 1,  # seconds
    },
    "kucoin":
            {
                "limit":1500,
                "pause_every":1,
                "pause":3
        },
    "huobi": {
        "limit": 1_000,
        "pause_every": 10,
        "pause": 1,  # seconds
    }
}

print(toml.dumps(config))