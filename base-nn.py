import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):

    def __init__(self, layers = []):
        # initialize nn.Module
        super(Net, self).__init__()

        # define model
        self.n_layers = n_layers = len(layers)
        if self.n_layers < 2:
            raise ValueError(f"Model must have at least 2 layers (input to output); now has {n_layers}")
        
        # initialize layers
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        """ feed forward a given input through 2 layers """
                
        # feed forward
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    
class SimpleNeuralNetwork():
    def __init__(self, features: torch.Tensor, targets: torch.Tensor, layers = [20, 10, 10, 1]):
        # validate data
        if not ( isinstance(features, torch.Tensor) and isinstance(targets, torch.Tensor) ):
            raise TypeError("both features and targets must be of type torch.Tensor")
        
        # store data and init model
        self.features, self.targets = features, targets
        self.model = Net(layers)
        
        print(self.model)
        
        
    def fit(self, epochs, show_accuracy = False):
        
        # initialize model and define loss function, optimizer
        # LATER PASS THESE AS PARAMETERS, ALSO FOR GRID SEARCH
        
        trainset = torch.utils.data.DataLoader(self.features, batch_size = 10, shuffle = True)
        
        
        
        optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        for epoch in enumerate(range(epochs), 1):

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
            
                # set gradient of optimizer at zero
                optimizer.zero_grad() 
                
                # compute the forecast of the model
                output = self.model(self.features)
                
                # compute the loss
                loss = criterion(output, self.targets)
                loss.backward()
                
                # update parameters
                optimizer.step()
                
                print(f"--------epoch {epoch}--------")
                print(f"trainig loss: {loss}")
                
    def predict(self, features):
        return self.model(features)
    

def get_ticker_data(ticker = "MSFT"):
    import yfinance
    
    tick = yfinance.Ticker(ticker)
    return tick.history(period = "max", interval = "1d")

def test():
    # get data from yahoo finance, square to get volatility
    data = get_ticker_data("MSFT")
    ret = (data["Close"].apply(np.log).diff()) ** 2
    
    # create lagged values as features
    lags = 10
    features = pd.DataFrame({})
    for i in range(lags):
        features[f"lag_{i}"] = ret.shift(i)
    
    # load data in tensors
    features, targets = torch.tensor(features.values, dtype=torch.float32), torch.tensor(ret.values, dtype=torch.float32)
    s = SimpleNeuralNetwork(features, targets, layers = [lags, 5, 1])
    s.fit(epochs = 10)
    s.predict(features)

test()