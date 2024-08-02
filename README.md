# Moving Average Distance as a Predictor of Cryptocurrency Returns [article](https://deliverypdf.ssrn.com/delivery.php?ID=017096020116090003008025004027089098015051021063079059125002022095123124109009068127006055125009047120052112024114022099020070052085021064086098089091093103003064028045087117116007001084102003115127025085064011116083072069031126083102065085098071027&EXT=pdf&INDEX=TRUE)
## Overview
This repository contains code and research findings related to the paper titled "Moving Average Distance as a Predictor of Equity Returns" by Doron Avramova, Guy Kaplanski, and Avanidhar Subrahmanyam. The research explores the predictive power of the distance between short- and long-run moving averages of prices (MAD) in the cross-section of equity returns.

## Idea
Imagine you have 10 stocks, how to create a strategy given this strategy, you can compute 10 different ratio mobing average and then to do the mean and use this mean ratio moving average to buy or to sell.
This idea was implemented that does not work well that means for each stock there is a unique combinaison, it is mainly caused by the fact each asset has its own standard deviation. For instance, if the std variation of BTC is two times lower than MATIC that means the ratio of the same X,Y moving average is quite different, X,Y for each cryptocurrency is dependant from the std deviation.
## Abstract
The MAD-based trading strategy demonstrates robust investment payoffs, with annualized value-weighted alphas around 9%. The predictability extends beyond other well-known anomalies, such as momentum, 52-week highs, and profitability. The strategy remains viable even after accounting for reasonable trading costs, and its effectiveness is stronger on the long side compared to the short side.

# Aim repository : does MAD allow to establish a wealthy strategy for cryptocurrencies ?

This repository contains code specifically tailored for applying the MAD strategy to cryptocurrency datasets. The code includes calculations of MAD, implementation of cryptocurrency trading strategies, and an examination of MAD's predictive power in the cryptocurrency market.

# Answer : It may possible to establish a wealthy strategy for cryptocurrencies.
## Example 1 :
<img src="img\example_1.jpg" width="1024"/>

## Example 2 :
<img src="img\example_2.jpg" width="1024"/>

# How to launch the program

If you are new launch :
```
./init.sh
```
If missing, add the execution permission to the shell file:

```
sudo chmod +x init.sh
```

It is needed to install [pandas](https://github.com/pandas-dev/pandas), [Technical Analysis](https://github.com/bukosabino/ta), [optuna](https://github.com/optuna/optuna), matplotlib

Download BTC/USDT,ETH/USDT,ADA/USDT and BNB/USDT for 1h,4h,1d thanks to ``src\utils\downloader.py``

Launch this command : 
```
python3 multi_optimiser_momentum_strategy.py
```



# Core Repository Structure
- data ( folder containing your data)
- experience (folder which is personal cointaing the results of different studies/strategies)
- img ( folder containing images for the README.md)
- src
    - [strategy](https://github.com/Dante-Berth/MAD_BTC/tree/main/src/strategy) this folder is dedicated to trading strategies you need to add your strategies in this folder
    - [strategy utils](https://github.com/Dante-Berth/MAD_BTC/tree/main/src/strategy_utils) this folder refers to main scripts used for computing/saving/optimizing a strategy or possible strategies 
    - utils
- study (folder which is personal containing your studies)
# Contribution
We developed a method to compare trading strategies. To compare two strategies we can divide strategy A by strategy B we can compare it as BTC/USDT for instance. Given this ratio, we can apply technocal indicators to observe, to compute how a method outperforms another method according the time. 
A good indicator for measuring the performance the [RSI](https://fr.wikipedia.org/wiki/Relative_strength_index) :

indeed if the RSI is higher than 50 than means the strategy placed to the numerator outperformed the strategy placed to the denominator. 
## Example 1 :
<img src="img\example_1_RSI.png" width="1024"/>

## Example 2 :
<img src="img\example_2_RSI.png" width="1024"/>

## Example 3 :
<img src="img\example_3_RSI.png" width="1024"/>

## Example 4 :
<img src="img\example_4_RSI.png" width="1024"/>

# Todo
WALK FORWARD OPTMIZATION IN BACKTESTING
CROSS-VALIDATION IN BACKTESTING
COMPUTE THE SHARP RATIO 
COMPUTE THE MAX DRAWDOWN
BETTER PLOT
# Images of Clustering
## Example 1
<img src="robustness_plot\Ratio_method_x_y_study_momentum_strategy_ADAUSDT_binance_15m.png" width="1024"/>

## Example 2
<img src="robustness_plot\Ratio_method_x_y_study_momentum_strategy_ADAUSDT_binance_1d.png" width="1024"/>

## Example 3
<img src="robustness_plot\Ratio_method_x_y_study_momentum_strategy_ADAUSDT_binance_1h.png" width="1024"/>

## Example 4
<img src="robustness_plot\Ratio_method_x_y_study_momentum_strategy_ADAUSDT_binance_4h.png" width="1024"/>

## Example 5
<img src="robustness_plot\Ratio_method_x_y_study_momentum_strategy_BNBUSDT_binance_15m.png" width="1024"/>

## Example 6
<img src="robustness_plot\Ratio_method_x_y_study_momentum_strategy_BNBUSDT_binance_1d.png" width="1024"/>

## Example 7
<img src="robustness_plot\Ratio_method_x_y_study_momentum_strategy_BNBUSDT_binance_1h.png" width="1024"/>

## Example 8
<img src="robustness_plot\Ratio_method_x_y_study_momentum_strategy_BNBUSDT_binance_4h.png" width="1024"/>

## Example 9
<img src="robustness_plot\Ratio_method_x_y_study_momentum_strategy_BTCUSDT_binance_15m.png" width="1024"/>

## Example 10
<img src="robustness_plot\Ratio_method_x_y_study_momentum_strategy_BTCUSDT_binance_1d.png" width="1024"/>

## Example 11
<img src="robustness_plot\Ratio_method_x_y_study_momentum_strategy_BTCUSDT_binance_1h.png" width="1024"/>

## Example 12
<img src="robustness_plot\Ratio_method_x_y_study_momentum_strategy_BTCUSDT_binance_4h.png" width="1024"/>

## Example 13
<img src="robustness_plot\Ratio_method_x_y_study_momentum_strategy_DOGEUSDT_binance_15m.png" width="1024"/>

## Example 14
<img src="robustness_plot\Ratio_method_x_y_study_momentum_strategy_DOTUSDT_binance_15m.png" width="1024"/>

## Example 15
<img src="robustness_plot\Ratio_method_x_y_study_momentum_strategy_ETHUSDT_binance_15m.png" width="1024"/>

## Example 16
<img src="robustness_plot\Ratio_method_x_y_study_momentum_strategy_ETHUSDT_binance_1d.png" width="1024"/>

## Example 17
<img src="robustness_plot\Ratio_method_x_y_study_momentum_strategy_ETHUSDT_binance_1h.png" width="1024"/>

## Example 18
<img src="robustness_plot\Ratio_method_x_y_study_momentum_strategy_ETHUSDT_binance_4h.png" width="1024"/>

## Example 19
<img src="robustness_plot\Ratio_method_x_y_study_momentum_strategy_MATICUSDT_binance_15m.png" width="1024"/>

## Example 20
<img src="robustness_plot\Ratio_method_x_y_study_momentum_strategy_OCEANUSDT_binance_15m.png" width="1024"/>

## Example 21
<img src="robustness_plot\Ratio_method_x_y_study_momentum_strategy_SOLUSDT_binance_15m.png" width="1024"/>

## Example 22
<img src="robustness_plot\Ratio_method_x_y_study_momentum_strategy_TRXUSDT_binance_15m.png" width="1024"/>

## Example 23
<img src="robustness_plot\Ratio_method_x_y_study_momentum_strategy_XRPUSDT_binance_15m.png" width="1024"/>


# Video MP4



# New feature added 
We recently added envelop strategy
