import numpy as np
from copy import deepcopy

import torch.utils
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, train_test_split

from statsmodels.regression.linear_model import OLS

from utils.functions import reset_model_weights
from utils.preprocessing import DataSet, DataSetNump
from utils.modelbuilder import ForwardNeuralNetwork

def model_estimator(
    model: nn.Module, 
    optimizer: optim.Optimizer, 
    criterion, 
    epochs: int, 
    trainloader: DataLoader, 
    testloader: DataLoader = None, 
    earlystopper = None,
    verbose: int = 0
    ):
    """
    This functions trains a neural network on training data, through gradient descent.
    I.e. for a number of epochs, go through all batches in the trainloader.
    For each batch a prediction is made, which is used to compute the gradient of the loss function for each parameter in the model.
    After each batch, the parameters are update through this gradient, multiplied by some loss function. 
    Some advanced methods like Adam can also be used, which still use the above explained steps as a basis.
    """
    
    for epoch in range(1, epochs+1):
        
        running_loss = []
        for data in trainloader:
            
            # get features and targets from the mini batch
            batch_features, batch_targets = data
        
            # reset gradient optimizer
            optimizer.zero_grad()
            
            # with the batch features, predict the batch targets
            output = model(batch_features)
            
            # compute the loss and .backward() computes the gradient of the loss function
            loss = criterion(output, batch_targets)
            loss.backward()
            
            # update parameters (something like params += -learning_rate * gradient)
            optimizer.step()
            
            # keep track of loss to log improvements of the fit
            running_loss += [loss.item()]

        # the in-sampe loss after each epoch
        train_running_loss = np.average(running_loss)

        # if testloader provided, compute the out-of-sample performance of the model
        if testloader:
            test_running_loss = model_evaluater(model, testloader, criterion)
            if verbose > 0:
                print("validation loss: {np.average(running_loss)}")
            
            # if an early stopper is provided, check if validation loss is still improving
            # if not, stop the estimation
            if earlystopper:
                if earlystopper.early_stop(test_running_loss):
                    break
    
    # at the end of the estimation, return the last loss on the validation data
    if testloader:
        return test_running_loss, epoch
    return train_running_loss
                    
                
def model_evaluater(model: nn.Module, data, criterion):
    """ given a model and dataloader or dataset, a prediction is made and average loss per batch (based on provided criterion function) is returned """    
    running_loss = []
    
    # if a dataset is provided, we don't have to iterate through all batches
    if isinstance(data, DataSet):
        output = model(data.x_t)
        return criterion(output, data.y_t)
        
    # for all batches
    for batch in data:
        batch_features, batch_targets = batch
        
        # predict
        output = model(batch_features.float())
        
        # compute loss
        loss = criterion(output, batch_targets)
        
        # store loss
        running_loss += [loss.item()]
        
    # return the average loss
    return np.average(running_loss)
                
class EarlyStopper():
    """ This class checks if the loss function is still improving by certain criteria, specified at initialization of this class"""
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
    
    def reset(self, ):
        self.counter = 0

def kfolds_fit_and_evaluate_model(
    input_size: int,
    output_size: int,
    hidden_layers: list, 
    kfold: TimeSeriesSplit, 
    data: DataSet,
    lr: float, 
    epochs: int, 
    earlystopper: EarlyStopper = None,
    normalize_features: bool = False
    ):
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
    for train_index, test_index in kfold.split(data.x_t):

        # reset weights to start estimating from exactly the same initialization for each fold
        # reset_model_weights(model)
        model = ForwardNeuralNetwork(input_size, output_size, hidden_layers)
        if earlystopper: earlystopper.reset() # the early stopper must also be reset

        # split data into feature and target data for neural network
        data_train, data_test = data.split(train_index, test_index)
        
        # estimate the data
        loss = single_fit_and_evaluate_model(model, data_train, data_test, lr, epochs, earlystopper, normalize_features, return_prediction=False)
        score_nn.append(loss)
    
    average_kfold_score = np.average(score_nn)
    return average_kfold_score

def single_fit_and_evaluate_model(
    model: nn.Module, 
    data_train: DataSet,
    data_test: DataSet,
    lr: float,
    epochs: int, 
    earlystopper: EarlyStopper,
    normalize_features: bool = False,
    return_prediction: bool = False,
    batch_size: int = 20,
    ) -> float:
    """ Estimate a given neural network and return the criterion score on the validation data """
    # fit normalizer on train features and normalize both training and validation features
    if normalize_features:
        scaler = StandardScaler()
        features_train = scaler.fit_transform(features_train)
        features_validation = scaler.transform(features_validation)
        
    loader_train = DataLoader(data_train, batch_size = 20)
    loader_test = DataLoader(data_test, batch_size = 20)
            
    # initialize and estimate the model
    criterion = nn.MSELoss()
    loss, epoch = model_estimator(
        model,
        optimizer = optim.Adam(model.parameters(), lr = lr), 
        criterion = criterion, 
        epochs=epochs,
        trainloader=loader_train, 
        testloader=loader_test,
        earlystopper=earlystopper)
    
    # perform out of sample prediction
    if return_prediction:
        output = model(data_test.x_t)
        loss = criterion(output, data_test.y_t)
        return loss.item(), output
    return loss


def fit_and_evaluateHAR(
    model: OLS, 
    data_train: DataSetNump, 
    data_test: DataSetNump, 
    normalize_features: bool = False,
    ):
    """ this functions estimates the HAR model by simple OLS regression of 'todays' volatility on tomorrows'"""
    if normalize_features:
        scaler = StandardScaler()
        data_train.x = scaler.fit_transform(data_train.x)
        data_test.x = scaler.transform(data_test.x)
        
    model.__init__(endog=data_train.y, exog=data_train.x)
    
    res = model.fit()
    output = res.predict(data_test.x)

    loss = np.var(data_test.y - output)
    return loss, output