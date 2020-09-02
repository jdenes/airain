<h1>
    <img src="./resources/airain.png" alt="AIRAIN" height="50px">
</h1>

Airain is an entrepreneurial research project aiming at building an AI able to predict the price evolution of any commodity
(currency exchange rate, cryptocurrency, securities, etc.).
On top of that, this AI is able to trade according to its prediction to maximize gain.

![Python](https://img.shields.io/badge/python-v3.6+-green.svg)
![License](https://img.shields.io/badge/copyrights-reserved-red.svg)

## Features

Currently using Intrinio for data source and Trading 212 as broker.

**Current features :**
- data ingestion through Intrinio's API
- model training
- model backtesting
- order placing in Trading 212 using Selenium
- automated trader for data updating and opening/closing positions
- graphical interface (under development)


## Example


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

- An Intrinio account (API key, free to get)
- A Trading 212 account (credentials, free to get)

See `.cfg` files in `/resources/` to understand how to fill those information.

## Usage

All codes are stored in `/src/` and should be executed there.
The only interaction should be with `main.py`, where high-level commands are defined.
Comment or uncomment functions in `"__main__"` and execute:
```bash
cd /src/
python main.py
```


## Copyrights

All rights are reserved on this project, which means you cannot use it, modify it or redistribute it without explicit
permission from the author.
