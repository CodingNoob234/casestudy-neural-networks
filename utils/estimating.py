import numpy as np

import torch.utils
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, train_test_split

from utils.functions import reset_model_weights
from utils.preprocessing import data_to_loaders

def model_estimator(
    model: nn.Module, 
    optimizer: optim.Optimizer, 
    criterion, epochs: int, 
    trainloader: torch.utils.data.DataLoader, 
    testloader: torch.utils.data.DataLoader = None, 
    earlystopper = None,
    verbose: int = 0
    ) -> None:
    """
    Calling this functions estimates a neural network with a given optimizer, criterion, training/testing data among other variables
    """
    
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
            model_evaluater(model, testloader, criterion)
            if verbose > 0:
                print("validation loss: {np.average(running_loss)}")
            
            # if an early stopper is provided, check if validation loss is still improving
            # if not, stop the estimation  
            if earlystopper:
                if earlystopper.early_stop(np.average(running_loss)):
                    print(f"Early stopping due to no decrease in validation loss at epoch: {epoch}")
                    break
    
    # at the end of the estimation, return the last loss on the validation data
    return np.average(running_loss)
                    
                
def model_evaluater(model: nn.Module, dataloader: torch.utils.data.DataLoader, criterion):
    """ given a model and dataloader, a prediction is made and average loss per batch (based on provided criterion function) is returned """
    running_loss = []
    
    # for all batches
    for data in dataloader:
        batch_features, batch_targets = data
        
        # predict
        output = model(batch_features.float())
        
        # compute loss
        loss = criterion(output, batch_targets)
        
        # store loss
        running_loss += [loss.item()]

    return np.average(running_loss)
                            
                
class EarlyStopper():
    """ This class checks if the loss function is still improving by certain criteria, specified at initialization """
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
    
    
###############################################################################################################
###############################################################################################################
###############################################################################################################
# MIGHT BE GOOD TO REWRIITE THIS FUNCTION INTO 2 FUNCTIONS, OR A CLASS INSTANCE

def kfolds_fit_and_evaluate_model(model: nn.Module, kfold: TimeSeriesSplit, features: np.ndarray, targets: np.ndarray, lr: float, epochs: int, normalize_features: bool = False) -> float:
    """ 
    This functions executes as number of steps:
    - initialise the model based on provided parameters
    - divide sample in several "folds" through time
    - for each kfold, get the training/testing data, normalise the data and put it into training/features dataloaders
    - estimate the model
    - predict the validation data and compute the accuracy
    """
    # initialise model with provided specification
    score_nn = []

    i = 0
    for train_index, test_index in kfold.split(features):

        # reset weights to start estimating from exactly the same initialization for each fold
        reset_model_weights(model)

        # split data into feature and target data for neural network
        features_train, features_validation, targets_train, targets_validation = features[train_index], features[test_index], targets[train_index], targets[test_index]
        
        # fit normalizer on train features and normalize both training and validation features
        if normalize_features:
            scaler = StandardScaler()
            features_train = scaler.fit_transform(features_train)
            features_validation = scaler.transform(features_validation)

        # all features and targets to float tensor
        features_train_tensor = torch.tensor(features_train, dtype=torch.float32)
        features_validation_tensor = torch.tensor(features_validation, dtype=torch.float32)
        targets_train_tensor = torch.tensor(targets_train, dtype=torch.float32)
        targets_validation_tensor = torch.tensor(targets_validation, dtype=torch.float32)
        
        # feature/target tensors to dataloaders
        trainloader, testloader = data_to_loaders(features_train_tensor, features_validation_tensor, targets_train_tensor, targets_validation_tensor)
                
        # initialize and estimate the model
        criterion = nn.MSELoss()
        loss = model_estimator(
            model,
            optimizer = optim.Adam(model.parameters(), lr = lr), 
            criterion = criterion, 
            epochs=epochs,
            trainloader=trainloader, 
            testloader=testloader,
            earlystopper= None )#EarlyStopper(patience = 3, min_delta = 0))
        
        # # perform out of sample prediction
        # output = model(features_validation_tensor)
        # loss = criterion(output, targets_validation_tensor)
        
        score_nn += [loss.item()]
    return np.mean(score_nn)