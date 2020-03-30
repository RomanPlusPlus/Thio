# Thio
Thio - a playground for real-time anomaly detection.

-------

What if you could add an anomaly to your streaming data by clicking a button, and immediatelly see how your awesome AI detects it?

With Thio, you can do it. 

## Data

Thio takes synthetic or real-life data as the input (you can toggle between them in config/settings.xml).

The synthetic data, if plotted, looks like something one would get from noisy physical sensors (see the picture below).

The real-life data are current cryptocurrencies exchange rates, fetched from [CoinGecko](https://www.coingecko.com/en).  

The data could have an arbitrary number of channels (defined in config/data_channels.xml)

![Alt text](Thio.png?raw=true "Title")

## Algos

Currently, Thio supports one anomaly detection algo - KitNET. It is a lightweight online anomaly detection algorithm, which uses an ensemble of [autoencoders](https://en.wikipedia.org/wiki/Autoencoder). KitNET was developed by Mirsky et al, 2018, and [released](https://github.com/ymirsky/KitNET-py) under MIT license. Please support them by citing their paper:

*Yisroel Mirsky, Tomer Doitshman, Yuval Elovici, and Asaf Shabtai, "Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection", Network and Distributed System Security Symposium 2018 (NDSS'18)* 

They're not affiliated with the Thio project. 

## Compute

You can run Thio on CPU, and it will still be able to detect anomalies in real-time, and learn on new data.

##  Installation and usage

Thio was tested on Ubuntu 16.04.

#### 0. Dowload

Download this repo.

#### 1. Setup and activate a conda virtual environment

Create the environment:

```conda create --name thio_env```

Type "y" and enter.

Activate it:

```conda activate thio_env```

Install dependencies: 

```conda install scipy pandas matplotlib```

#### 2. Launch 

Launch 0launcher.py and wait a few sec. 

A window (similar to the one depicted above) will be opened, with plots regularly updating.

Click the "add an anomaly" button and observe a bump in the bottom graph. 

The anomaly was added to the input data, the data was processed by the anomaly-detection algo, and the corresponding risk scores were plotted.   


##  Name 

Thio is named after Isaac Asimov's [Thiotimoline](https://en.wikipedia.org/wiki/Thiotimoline), a fictitious chemical that starts dissolving **before** it makes contact with water:) 
