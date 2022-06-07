<h1>
    <img src="./resources/airain.png" alt="AIRAIN" height="50px">
</h1>

Airain is an research project aiming at building an AI able to suggest optimal portfolio allocation (currently for stocks and cryptocurrencies).
On top of that, this AI is able to open and close its suggested positions.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

Currently using Yahoo Finance for data source and Trading 212 as broker.

**Current features :**
- data ingestion using Yahoo Finance (code for Intrinio's API are also proposed)
- model training
- model backtesting
- order placing in Trading 212 using Selenium
- automated trader for data updating and opening/closing positions
- graphical interface using Dash (under development)


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

Rights are granted to anyone under a standard MIT license without any limitation (which means you can do whatever you want with it).
Yet I'd love to know if this repo was of any use to you!
