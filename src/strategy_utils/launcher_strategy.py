import os
import sys
absolute_path = os.path.abspath(__file__)[:os.path.abspath(__file__).find("MAD_BTC")+len("MAD_BTC")]
sys.path.append(absolute_path)

class trading_strategy:
    def __init__(self,config) -> None:
        self.config = config
    def strategy_rules():
        raise "To be implemented"
    def renderer():
        raise "To be implemented"