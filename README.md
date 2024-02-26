# Moving Average Distance as a Predictor of Cryptocurrency Returns [article](https://deliverypdf.ssrn.com/delivery.php?ID=017096020116090003008025004027089098015051021063079059125002022095123124109009068127006055125009047120052112024114022099020070052085021064086098089091093103003064028045087117116007001084102003115127025085064011116083072069031126083102065085098071027&EXT=pdf&INDEX=TRUE)
## Overview
This repository contains code and research findings related to the paper titled "Moving Average Distance as a Predictor of Equity Returns" by Doron Avramova, Guy Kaplanski, and Avanidhar Subrahmanyam. The research explores the predictive power of the distance between short- and long-run moving averages of prices (MAD) in the cross-section of equity returns.

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
It is needed to install [pandas](https://github.com/pandas-dev/pandas), [Technical Analysis](https://github.com/bukosabino/ta), [optuna](https://github.com/optuna/optuna), matplotlib

Download BTC/USDT,ETH/USDT,ADA/USDT and BNB/USDT for 1h,4h,1d thanks to ``src\utils\downloader.py``

Launch this command : 
```
python3 multi_optimiser_momentum_strategy.py
```


