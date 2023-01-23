import numpy as np
import pandas as pd

import torch.nn as nn

def print_nicely(note, length):
    """ 
    Prints a note in the style of:
    ============
    ==={note}===
    ============
    ... but with a given length
    """
    length = 100
    l = int((length-len(note))/2)
    m = length - l - len(note)
    print("="*length)
    print(l*"=" + note + m*"=")
    print("="*length)

def get_rv_from_yahoo(ticker = "MSFT"):
    """ returns the daily close price for a maximum period for specified ticker """
    import yfinance
    
    tick = yfinance.Ticker(ticker)
    price = tick.history(period = "max", interval = "1d")["Close"]
    return price.apply(np.log).diff()**2

def get_rv_from_data(ticker = "AAPL"):
    df = pd.read_excel("data/data_2015-2023.xlsx")
    df.columns = ("date", "ticker", "ticker2", "vol_realized")
    df = df[["date", "ticker", "vol_realized"]]
    try:
        return df[df["ticker"] == ticker]["vol_realized"].reset_index(drop=True)
    except:
        raise Exception("Ticker not found in data")
    
def get_tickers_in_data():
    df = pd.read_excel("data/data_2015-2023.xlsx")
    df.columns = ("date", "ticker", "ticker2", "vol_realized")
    return list(df["ticker"].unique())

def reset_model_weights(m: nn.Module):
    """ Resets all weights of the neural network to those at initialization """
    for layer in m.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()