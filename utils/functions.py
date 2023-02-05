import pandas as pd
import torch.nn as nn

def get_rv_from_yahoo(ticker = "MSFT"):
    """ returns the daily estimated realized volatility, by squaring the daily returns """
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
    """ This functions plots realized volatility for all indices available in the datafile """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # get RV for each individual ticker from the data and store in dict
    tickers = get_tickers_in_data()
    rvs = [get_rv_from_data(ticker) for ticker in tickers]
    rvs = dict(zip(tickers, rvs))
    
    # create figure and plot
    w, h = 4, int(np.ceil(len(tickers) / 4))
    fig, axs = plt.subplots(h, w, figsize = (15, 15))
    for i, ticker in enumerate(tickers):
        axs[int(i//w), int(i%w)].plot(rvs[ticker])
        axs[int(i//w), int(i%w)].set_title(ticker)
    plt.show()

def reset_model_weights(m: nn.Module):
    """ Resets all weights of a neural network to those at initialization """
    for layer in m.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()