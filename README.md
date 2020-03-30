# Thio
Thio - a playground for real-time anomaly detection.

-------

What if you could add an anomaly to your streaming data by clicking a button, and immediatelly see how your awesome AI detects it?

With Thio, you can do it. 

## Data

Thio takes synthetic or real-life data as the input (you can toggle between them in config/settings.xml).

The synthetic data, if plotted, looks like something one can get from noisy physical sensors (see the picture below).

The real-life data are current cryptocurrencies exchange rates, fetched from [CoinGecko](https://www.coingecko.com/en).  


![Alt text](Thio.png?raw=true "Title")

## Algos

Currently, Thio supports one anomaly detection algo - KitNET. It is a lightweight online anomaly detection algorithm, which uses an ensemble of [autoencoders](https://en.wikipedia.org/wiki/Autoencoder). KitNET was developed by Mirsky et al, 2018, and [released](https://github.com/ymirsky/KitNET-py) under MIT license. Please support them by citing their paper:

*Yisroel Mirsky, Tomer Doitshman, Yuval Elovici, and Asaf Shabtai, "Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection", Network and Distributed System Security Symposium 2018 (NDSS'18)* 

They're not affiliated with the Thio project. 
