import toml
import datetime

config = {
    "since": datetime.datetime.now().isoformat(),
    "until": datetime.datetime(year=2018, month=4, day=1).isoformat(),
}

print(toml.dumps(config))

# load a config file:
with open("src/configs/download_config.toml", "r") as file:
    config = toml.load(file)

date_str = config.get("since")
date = datetime.datetime.fromisoformat(date_str)
print(date_str)
print(date)
