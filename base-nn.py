import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.Functional as F
import torch.optim as optim

class Net(nn.Module):

    def __init__(self):
        # initialize nn.Module
        super(Net, self).__init__()

        # two simple layers operation: y = Wx + b
        self.fc1 = nn.Linear(10, 5) 
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        """ feed forward a given input through 2 layers """
        # x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
      
# features and targets
features, targets = torch.load #....    

# initialize model and define loss function, optimizer
net = Net()
optimizer = optim.SGD(net.parameters(), lr=0.01)
criterion = nn.MSELoss()

# estimate the model
epochs = 10

show_accuracy = False

for epoch in enumerate(1, epochs):
  
    # set gradient of optimizer at zero
    optimizer.zero_grad() 
    
    # compute the forecast of the model
    output = net(input)
    
    # compute the loss function
    loss = criterion(output, target)
    loss.backward()
    
    # update parameters
    optimizer.step()    # Does the update
    
    if show_accuracy:
        