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
    def __init__(self, lags, nodes = [], hidden_dim = 1, n_layers = 1, output_size = 1):
        # initialize nn.Module
        super(Net, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # initialize layers
        self.rn = nn.RNN(lags, hidden_dim, n_layers, batch_first = True)
        self.fc = nn.Linear(hidden_dim, output_size)
        
        # self.fc1 = nn.Linear(lags, nodes[0], bias = True)
        # self.fc2 = nn.Linear(nodes[0], 1, bias = True)

    def forward(self, x: torch.Tensor):
        """ feed forward a given input through 2 layers """
             
        # x = torch.flatten(x,1)
        # x = F.sigmoid(self.fc1(x))
        # x = self.fc2(x)
        
        #==================================
        
        x = x.unsqueeze(0)
        batch_size = x.size(0)
        
        hidden = self.init_hidden(batch_size)
        x, hidden = self.rn(x, hidden)
        x = x.contiguous().view(-1, self.hidden_dim)
        
        x = self.fc(x)
        return x
    
    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
    
class SimpleNeuralNetwork():
    """
    This model suits as an extra layer on top of the model.
    This way, for estimating a certain model, we only need to call
    
    mod = SimpleNeuralNetwork(features, targets)
    mod.fit()
    mod.predict(features_test)
    """
    
    def __init__(self, lags, nodes, hidden_dim = 1, n_layers = 1, output_size = 1):
        # init model params
        self.model = Net(lags, nodes, hidden_dim, n_layers, output_size)
        
    def fit(self, 
            trainloader: torch.utils.data.DataLoader, 
            testloader: torch.utils.data.DataLoader = None,
            epochs: float = 10, 
            lr: float = 0.01, 
        ):
        
        # define optimizer and loss function   (PASS AS PARAMS LATER?)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = criterion = nn.MSELoss()

        for epoch in range(1, epochs+1):

            running_loss = []
            for data in trainloader:
                
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
                running_loss += [loss.item()]
                
            avg_running_loss = np.average(running_loss)
            
            # print(f"epoch: {epoch}")
            # print(f"trainig loss: {avg_running_loss}")
            if testloader:
                running_loss = []
                for data in testloader:
                    batch_features, batch_targets = data
                    output = self.model(batch_features.float())
                    loss = criterion(output, batch_targets)
                    loss.item()
                    running_loss += [loss.item()]
                # print("validation loss: {np.average(running_loss)}")
    
    def __call__(self, features: torch.Tensor):
        return self.model(features)