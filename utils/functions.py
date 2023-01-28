import pandas as pd
import numpy as np
import torch.nn as nn
import math

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
    import numpy as np
    
    tick = yfinance.Ticker(ticker)
    price = tick.history(period = "max", interval = "1d")["Close"]
    return price.apply(np.log).diff()**2

def get_rv_from_data(ticker = ".AEX"):
    df = pd.read_csv("data/oxfordmanrealizedvolatilityindices.csv")
    try:
        return df[df.Symbol == ticker]["rv5"].reset_index(drop=True)
    except Exception as e:
        raise Exception("Ticker not found in data")
    
def get_tickers_in_data():
    """ return all tickers available in the realized volatility dataset """
    df = pd.read_csv("data/oxfordmanrealizedvolatilityindices.csv")
    return list(df.Symbol.unique())

def plot_tickers_in_data():
    import matplotlib.pyplot as plt
    tickers = get_tickers_in_data()
    rvs = [get_rv_from_data(ticker) for ticker in tickers]
    rvs = dict(zip(tickers, rvs))
    w = 4
    h = int(np.ceil(len(tickers) / 4))
    fig, axs = plt.subplots(h, w, figsize = (10, 10))
    for i, ticker in enumerate(tickers):
        axs[int(i//w), int(i%w)].plot(rvs[ticker])
    plt.show()

def reset_model_weights(m: nn.Module):
    """ Resets all weights of the neural network to those at initialization """
    for layer in m.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()