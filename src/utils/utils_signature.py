import json
import hashlib

def create_signature(*args,**kwargs):
    json_data = json.dumps(*args,**kwargs, sort_keys=True).encode('utf-8')
    signature = hashlib.sha256(json_data).hexdigest()
    return signature

if __name__=="__main__":
    config = {
    "crypto_currency":"ETHUSDT",
    "time":"4h",
    "exchange":"binance",
    "x":12,
    "y":90,
    "ratio_method":"ema_ratio",
    "initial_capital":1000
    
    }
    print(create_signature(config))

    config =  {
    "crypto_currency":"ETHUSDT",
    "time":"4h",
    "exchange":"binance",
    "strategy":{
        "x":12,
        "y":90,
        "ratio_method":"ema_ratio"
    },
    "initial_capital":1000
    
    }
    print(create_signature(config))
    print(create_signature(config["strategy"]))