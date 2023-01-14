import numpy as np

import torch.utils
import torch.optim as optim
import torch.nn.functional as F

def model_estimator(model, optimizer, criterion, epochs, trainloader, testloader = None, verbose = 0):
    for epoch in range(1, epochs+1):
        
        running_loss = []
        for data in trainloader:
            
            # get features and targets from the mini batch
            batch_features, batch_targets = data
        
            # reset gradient optimizer
            optimizer.zero_grad()
            
            # predict the batch targets
            output = model(batch_features)
            
            # compute the loss and .backward() computes the gradient respective to the loss function
            loss = criterion(output, batch_targets)
            loss.backward()
            
            # update parameters (something like params += -learning_rate * gradient)
            optimizer.step()
            
            # keep track of loss to log improvements of the fit
            running_loss += [loss.item()]
            
        avg_running_loss = np.average(running_loss)

        # if verbose, print intermediate model performances in and out of sample
        if verbose > 1:        
            print(f"epoch: {epoch}")
            print(f"trainig loss: {avg_running_loss}")
        if testloader:
            running_loss = []
            for data in testloader:
                batch_features, batch_targets = data
                output = model(batch_features.float())
                loss = criterion(output, batch_targets)
                loss.item()
                running_loss += [loss.item()]
            if verbose > 0:
                print("validation loss: {np.average(running_loss)}")