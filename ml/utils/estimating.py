import numpy as np

import torch.utils
import torch.optim as optim
import torch.nn.functional as F

def model_estimator(model, optimizer, criterion, epochs, trainloader, testloader=None, verbose = 0):
    
    earlystopper = EarlyStopper(patience = 3, min_delta = 0)
    
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
            
            # if validation results are not improving    
            if earlystopper.early_stop(np.average(running_loss)):
                print(f"early stopping due to no decrease in validation loss at epoch: {epoch}")
            
                
class EarlyStopper():
    def __init__(self, patience = 1, min_delta = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
        
    def early_stop(self, validation_loss):
        """ check if the validation loss has not increased for {patience}-times """
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False