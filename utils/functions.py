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

def get_ticker_daily_close(ticker = "MSFT"):
    """ returns the daily close price for a maximum period for specified ticker """
    import yfinance
    
    tick = yfinance.Ticker(ticker)
    return tick.history(period = "max", interval = "1d")["Close"]

def reset_model_weights(m: nn.Module):
    """ Resets all weights of the neural network to those at initialization """
    for layer in m.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()