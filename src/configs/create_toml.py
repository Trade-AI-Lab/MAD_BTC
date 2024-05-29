import toml
import datetime

config = {"date": datetime.datetime.now().isoformat()}

print(toml.dumps(config))

# load a config file:
with open("src/configs/download_config.toml", "r") as file:
    config = toml.load(file)

date_str = config.get("date")
date = datetime.datetime.fromisoformat(date_str)
print(date_str)
print(date)
