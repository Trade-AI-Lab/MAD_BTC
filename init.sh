git clone https://github.com/Dante-Berth/MAD_BTC.git
cd MAD_BTC
python -m venv trading_research
source trading_research/bin/activate
pip install -r requirements.txt
python3 multi_optimiser_momentum_strategy.py