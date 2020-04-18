# Thio
Thio - a playground for real-time anomaly detection.

-------

What if you could add an anomaly to your streaming data by clicking a button, and immediatelly see how an AI detects it?

With Thio, you can do it. 

## Data

Thio takes synthetic or real-life data as the input (you can toggle between them in config/settings.xml).

The synthetic data is generated on the fly. If plotted, it looks like something one would get from noisy physical sensors (see the picture below).

The real-life data are current cryptocurrencies exchange rates, fetched from [CoinGecko](https://www.coingecko.com/en).  

The data could have an arbitrary number of channels (defined in config/data_channels.xml)

![Alt text](Thio.png?raw=true "Title")

## Algos

Currently, Thio supports two anomaly detection algos - KitNET and Telemanom. Both are ANN-based, and both learn in an unsupervised manner. 

**KitNET** is a lightweight online anomaly detection algorithm, which uses an ensemble of [autoencoders](https://en.wikipedia.org/wiki/Autoencoder). KitNET was developed by Mirsky et al, 2018, and [released](https://github.com/ymirsky/KitNET-py) under MIT license. Please support them by citing their [paper](https://arxiv.org/abs/1802.09089):

*Mirsky, Doitshman, Elovici, Shabtai. "Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection", Network and Distributed System Security Symposium 2018 (NDSS'18)* 

**Telemanom** is a framework for using [LSTMs](https://en.wikipedia.org/wiki/Long_short-term_memory) to detect anomalies in multivariate time series data. It was developed by Hundman et al, 2018, and [released](https://github.com/khundman/telemanom) under an Apache 2.0 license. Please support them by citing their [paper](https://arxiv.org/abs/1802.04431):

*Hundman, Constantinou, Laporte, Colwell, Soderstrom. "Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding". KDD '18: Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, July 2018, Pages 387–395*

None of the mentioned people or organisations are affiliated with the Thio project. 

## Compute

You can run Thio on CPU, and it will still be able to detect anomalies in real-time, and learn on new data. 

Training and inference processes are running in parallel. The models are regularly retrained on the latest N datapoints. The inference process uses the most recent model to produce risk scores.

##  Installation and usage

Thio was tested on Ubuntu 16.04, with the following packages versions:
* conda==4.8.3
* Python==3.8.2 (in the thio_kitnet env)
* Python==3.7.7 (in the thio_telemanom env)
* numpy==1.18.1
* pandas==1.0.3
* pyyaml==5.3.1
* keras==2.3.1
* tensorflow==2.1.0
* theano==1.0.4
* cufflinks==0.17.3
* more_itertools==8.2.0
* scipy==1.4.1
* matplotlib==3.1.3
* requests==2.23.0
* psutil==5.7.0

#### 0. Dowload

Download this repo.

#### 1. Setup the first virtual environment

Create it:

```conda create --name thio_telemanom```

Type "y" and enter.

Activate it:

```conda activate thio_telemanom```

Install dependencies: 

```conda install numpy pandas pyyaml keras```

```conda install -c conda-forge tensorflow```

```pip install theano cufflinks more_itertools```

Deactivate it:

```conda deactivate```


#### 2. Setup the second virtual environment

Create it:

```conda create --name thio_kitnet```

Type "y" and enter.

Activate it:

```conda activate thio_kitnet```

Install dependencies: 

```conda install scipy pandas matplotlib requests psutil```



#### 3. Launch 

cd to the dir where 0launcher.py is located.

Launch 0launcher.py and wait a few sec. 

A window (similar to the one depicted above) will be opened, with plots regularly updating.

Click the "add an anomaly" button and observe a bump in the two bottom graphs. 

The anomaly was added to the input data, the data was processed by the anomaly-detection algo, and the corresponding risk scores were plotted.   

### 4. Tips

If you want a KitNET model that can produce meaningful results, you need a dataset of at least 30 000 datapoints (a thumb rule). Same for Telemanom.

Telemanom will take some time after the launch before starting to detect anomalies. On my modest hardware (no GPU inference), it's about 15 min. 

### 5. Name 

Thio is named after Isaac Asimov's [Thiotimoline](https://en.wikipedia.org/wiki/Thiotimoline), a fictitious chemical that starts dissolving **before** it makes contact with water:) 
