import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class Net(nn.Module):
    """ This class holds our model with 2 layers """
    def __init__(self, lags, nodes = []):
        # initialize nn.Module
        super(Net, self).__init__()
    
        assert len(nodes) == 3, "length nodes must be equal to number of layers"
    
        # initialize layers
        self.fc1 = nn.Linear(lags, nodes[0])
        self.fc2 = nn.Linear(nodes[0], nodes[1])
        self.fc3 = nn.Linear(nodes[1], 1)

    def forward(self, x):
        """ feed forward a given input through 2 layers """
                
        # feed forward
        # x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    
class SimpleNeuralNetwork():
    """
    This model suits as an extra layer on top of the model.
    This way, for estimating a certain model, we only need to call
    
    mod = SimpleNeuralNetwork(features, targets)
    mod.fit()
    mod.predict(features_test)
    """
    
    def __init__(self, lags, nodes):
        # init model params
        self.model = Net(lags, nodes)
        
    def fit(self, data, 
            features_val: torch.Tensor = None, 
            targets_val: torch.Tensor = None, 
            epochs: float = 10, 
            lr: float = 0.01, 
            batch_size: int = 50
        ):
        
        # validate data
        trainloader = torch.utils.data.DataLoader(data, batch_size = batch_size, shuffle = True)
        
        optimizer = optim.SGD(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        # criterion = nn.NLLLoss()

        for epoch in range(1, epochs + 1):

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                
                # get features and targets from the mini batch
                batch_features, batch_targets = data
            
                # reset gradient optimizer
                optimizer.zero_grad()
                
                # predict the batch targets
                output = self.model(batch_features)
                
                # compute the loss and .backward() computes the gradient respective to the loss function
                loss = criterion(output, batch_targets)
                loss.backward()
                
                # update parameters (something like params += -learning_rate * gradient)
                optimizer.step()
                
                # keep track of loss to log improvements of the fit
                running_loss += (loss.item()*batch_size)

            print("-" * 100)
            print(f"epoch: {epoch}")
            print(f"trainig loss: {running_loss}")
            if features_val and targets_val:
                output = self.model(features_val)
                loss = criterion(output, targets_val)
                print("validation loss: {loss.item()}")
                
    def predict(self, features):
        return self.model(features)
    
# def scale_data(train_data, test_data):
#     train_x, train_y = [], [train_data]
#     scaler = StandardScaler()
#     scaler.fit_transform()
#     return 
    

def get_ticker_data(ticker = "MSFT"):
    import yfinance
    
    tick = yfinance.Ticker(ticker)
    return tick.history(period = "max", interval = "1d")

def test():
    # get data from yahoo finance, square to get volatility
    data = get_ticker_data("MSFT")
    targets = (data["Close"].apply(np.log).diff()) ** 2
        
    # create lagged values as features
    lags = 10
    features = pd.DataFrame({})
    for i in range(1, lags + 1):
        features[f"lag_{i}"] = targets.shift(i)
    features = features.values.reshape(-1,lags)
    targets = targets.values.reshape(-1,1)
    
    print(f"The targets are of shape: {targets.shape}")
    print(f"The features are of shape: {features.shape}")
    
    # drop nan values
    features = features[lags+1:-max(1, int(.3*lags))]
    targets = targets[lags+1:-max(1, int(.3*lags))]
    
    # load data in tensors
    features, targets = torch.tensor(features, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)

    features, features_test, targets, targets_test = train_test_split(features, targets, train_size = .8)

    data = [(feature, target) for feature, target in zip(features, targets)]
    
    # init and estimate the model
    s = SimpleNeuralNetwork(lags, nodes = [160, 480, 256])
    s.fit(data, epochs = 10, lr = .1, batch_size = 20)
    
    # prediction = s.predict(features).detach().numpy()
    # plt.plot(targets.detach().numpy(), label = "target")
    # plt.plot(prediction, label = "pred")
    # plt.legend()
    # plt.ylim(0, 0.01)
    # plt.show()
    
test()