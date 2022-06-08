<h1>
    <img src="./resources/airain.png" alt="AIRAIN" height="50px">
</h1>

Airain is a research project aiming at building an AI able to suggest optimal portfolio allocation for day trading 
(currently for stocks and cryptocurrencies). On top of that, it is able to open and close its recommended portfolio.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-v2.4.1-orange?logo=tensorflow&logoColor=orange)](https://www.tensorflow.org/)

## Features

- 📡 data retrieval using Yahoo Finance API (code for Intrinio's API are also proposed)
- 🏋️ model training (see the architecture below)
- 🤺 model backtesting
- 🦾 automated, fast order placing in Trading 212 using Selenium for Firefox
- 🤖 background-running trader for automated data updating and opening/closing positions
- 📈 graphical interface using Dash (under development)


## Installation

#### Requires Python 3.6+.

Clone the repo : 
```bash
git clone https://github.com/jdenes/airain.git
```

Then install requirements:
```bash
pip install -r requirements.txt
```

#### Other requirements
A Trading 212 account and credentials (a free version allows you to trade using fake money).
See `trading212_example.cfg` files in `/resources/` to understand how to fill credentials. Yours should be place in a new
`trading212.cfg` file in the same folder.


## Usage

All codes are stored in `/src/` and should be executed there.
The only interaction should be with `main.py`, where high-level commands are defined.
Comment or uncomment functions in `"__main__"` and execute (better version with argparse to come):
```bash
cd /src/
python main.py
```


## Technical overview

This repo contains the result of more than two years of experiments. It consists of a final chosen model architecture, but also
a lot of unused material. We present then both here, with their designated files.

### Most promising model

Heavily inspired by the paper [Deep Learning for Portfolio Optimization](https://arxiv.org/pdf/2005.13665.pdf), the intuition
is to use a neural network which output shape is the number of asset plus one, representing the cash bias. The reward used
is the obtained return. As we chose to work under a supervised learning task and not a reinforced learning one, we use it as a
los function instead of a reward.

#### Data
Features are created in file `data_processing.py`. The best 98 where selected, and are very diverse: asset-specific values,
lags, analysis of the japanese market, time features, etc. The model chosen is based on a specific formatted matrix, with
shape *i* × *m* × *h* for *i* assets, with their *m* features, and a history of the past *h* days. Data is then heavily noised
using gaussian random noise, in order to prepare the model for "real-world" conditions.

#### Model
A custom model, which can be found in `traders.py`, is a neural network with the following layers:
- for each asset *i*, represented by its *m* × *h* matrix, if fed to a dense NN which outputs an array of size *i*;
- then the outputs of each asset-specific NN are concatenated;
- this array is fed to an *i+1* dense NN, with the *+1* representing the cash bias;
- finally a softmax activation function is applied to ensure that the output is a unit vector, summing up to 1.

#### Loss function
We use the achieved return of the portfolio on the next day, times -1, as a loss function. However, this loss exhibited poor behavior
it terms of risk: it almost always chose a single asset. To balance this, we add as a penalizing term the categorical crossentropy
between the proposed portfolio and the optimal one. Losses are implemented in `traders.py` as well.

### Other tested approaches
- interpreting the task as a regression task, and trying to predict the price of each asset for next day: attempts with NN and LightGBM: explore `traders.py`
- using Generative Adversarial Network (GAN) to create more training example: see `gan.py`
- analysing the press around each asset to predict price, either as such (BERT model trained to predict the price) or as a feature. Explore `data_processing.py`

### How does it perform?
The overall conclusion is that it is able to beat a benchmark (a portfolio that is evenly distributed among each stock), but not the best stock.
Moreover, it does not perform well under "real-world" tests, as it is very sensitive to inaccurate/noisy market values measurements.
Yes there is still lots of room for improvement!

## Copyrights

Rights are granted to anyone under a standard MIT license without any limitation (which means you can do whatever you want with it).
Yet I'd love to know if this repo was of any use to you, so don't hesitate to reach out!
