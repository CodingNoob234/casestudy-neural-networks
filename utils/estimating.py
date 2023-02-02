# standard import
import math
import numpy as np

# ML
import torch
import torch.utils
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, train_test_split

# fitting HAR
from statsmodels.regression.linear_model import OLS

# own functions
from utils.functions import reset_model_weights
from utils.preprocessing import DataSet, DataSetNump
from utils.modelbuilder import ForwardNeuralNetwork

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
        self.min_validation_loss = np.inf
        
    def __str__(self,):
        return f"EarlyStopper(patience={self.patience},min_delta={self.min_delta})"
        
def model_estimator(
    model: nn.Module, 
    optimizer: optim.Optimizer, 
    criterion, 
    epochs: int, 
    trainloader: DataLoader, 
    testloader = None, 
    earlystopper: EarlyStopper = None,
    verbose: int = 0,
    ):
    """
    This functions trains a neural network on training data, through gradient descent.
    In short, it will go through all batches in the trainloader for a number of epochs.
    For each batch a prediction is made, which is used to compute the gradient of the loss function, for each parameter in the model.
    After each batch, the parameters are update by this gradient, multiplied by some loss function. 
    Some advanced methods like Adam can also be used, which still the principals of the above explained method.
    
    Some parameters might need some elaboration:
    - criterion: must be a loss function from torch, with parent class nn.Module
    - testloader: may be a torch dataloader with batches, or DataSet instance that holds the features/targets NOT divided in batches
    - verbose: determines to what level intermediate results are be printed. Especially used in the early stage of building the model and code
    - early_stopper_on_test: whether or not to apply the earlystopper on test or training data
    """
    
    train_loss_list = []
    
    for epoch in range(1, epochs+1):
        
        # train_running_loss = []
        for batch in trainloader:
            
            # each batch contains one set of features and one set of targets
            batch_features: torch.Tensor = batch[0]
            batch_targets: torch.Tensor = batch[1]
        
            # reset gradient optimizer
            optimizer.zero_grad()
            
            # with the batch features, predict the batch targets
            output = model(batch_features)
            
            # compute the loss and .backward() computes the gradient of the loss function
            loss = criterion(output, batch_targets)
            loss.backward()
            
            # update parameters (something like: params += -learning_rate * gradient)
            optimizer.step()
            
            # keep track of loss to log improvements of the fit
            # train_running_loss.append(loss.item())

        # after each epoch evaluate on test or trainloader
        running_loss: float
        if testloader:
            with torch.no_grad():
                test_running_loss = model_evaluater(model, testloader, criterion)
                if verbose > 0:
                    print(f"epoch {epoch} - validation loss: {test_running_loss}")
                running_loss = test_running_loss
        else:
            with torch.no_grad():
                train_running_loss = model_evaluater(model, trainloader, criterion)
                train_loss_list.append(train_running_loss)
                if verbose > 0:
                    print(f"epoch {epoch} - training loss: {train_running_loss}")
                running_loss = train_running_loss

        # if an early stopper is provided, check if loss is still improving
        # if not, break the training loop
        if earlystopper:
            if earlystopper.early_stop(running_loss):
                break
    
    # after all epochs, or when early stopped, return the last loss on the validation data
    if testloader:
        return test_running_loss, epoch
    return train_running_loss, epoch
                    
                
def model_evaluater(model: nn.Module, data, criterion):
    """ given a model and dataloader or dataset, a prediction is made and average loss per batch (based on provided criterion function) is returned """    
    
    # if a dataset is provided, we don't have to iterate through all batches
    if isinstance(data, DataSet):
        output = model(data.x_t).detach().numpy().reshape(-1,1)
        resid = data.y_t.detach().numpy().reshape(-1,1) - output
        return np.average(resid**2)
        
    # if batches instead of one large dataset
    running_loss = []
    
    for batch in data:
        batch_features, batch_targets = batch
        
        # predict
        output = model(batch_features)
        
        # compute loss
        loss = criterion(output, batch_targets)
        
        # store loss
        running_loss += [loss.item()]
        
    # return the average loss
    return np.average(running_loss)

def kfolds_fit_and_evaluate_model(
    input_size: int,
    output_size: int,
    hidden_layers: list, 
    kfold: TimeSeriesSplit, 
    data: DataSet, 
    lr: float, 
    epochs: int, 
    earlystopper: EarlyStopper = None,
    normalize_features: bool = False,
    batch_size: int = 10,
    ):
    """ 
    This functions executes a number of steps:
    - divide sample in several "folds" through time, using TimeSeriesSplit
    - for each fold
        - initialise a neural network
        - get the training/testing data, normalise the data (if required) and put it into training/features dataloaders
        - train the model
        - evaluate on the test data from the fold and append to a list
    - return the average of all fold losses
    """
    # initialise model with provided specification
    score_nn = []

    for train_index, test_index in kfold.split(data.x_t):

        # instantiate a new model, so the weights are randomly initialized with a new seed
        model = ForwardNeuralNetwork(input_size, output_size, hidden_layers)
        if earlystopper: earlystopper.reset() # the early stopper must also be reset (counter, minimum loss)

        # split data into feature and target data for neural network
        data_train, data_test = data.split(train_index, test_index)
        
        # estimate the data
        loss = single_fit_and_evaluate_model(
            model = model, 
            data_train = data_train, 
            data_test = data_test, 
            lr = lr, 
            epochs = epochs, 
            earlystopper = earlystopper, 
            normalize_features = normalize_features, 
            return_prediction=False, 
            batch_size = batch_size
        )
        score_nn.append(loss)
    
    average_kfold_score = np.average(score_nn)
    return average_kfold_score

def single_fit_and_evaluate_model(
    model: nn.Module, 
    lr: float,
    epochs: int, 
    data_train: DataSet,
    data_test: DataSet = None,
    earlystopper: EarlyStopper = None,
    normalize_features: bool = False,
    return_prediction: bool = False,
    batch_size: int = 20,
    ):
    """ 
    Estimation of a neural network can be generalized by a few parameters.
    This function executes:
    - train/test data to train/test loaders (batches)
    - train model
    - predict the testing data and return the loss
    """
    # fit normalizer on train features and normalize both training and validation features
    if normalize_features:
        scaler = StandardScaler()
        features_train = scaler.fit_transform(features_train)
        features_validation = scaler.transform(features_validation)
     
    # NOTE: SHOULD SHUFFLE BE TRUE OR FALSE, WAS FALSE UNTIL NOW
    loader_train = DataLoader(data_train, batch_size = batch_size, shuffle = True)
    loader_test = DataLoader(data_test, batch_size = batch_size, shuffle = True) if data_test else None

    # initialize and train the model
    criterion = nn.MSELoss()
    loss, epoch = model_estimator(
        model,
        optimizer = optim.Adam(model.parameters(), lr = lr), 
        criterion = criterion, 
        epochs = epochs,
        trainloader = loader_train, 
        testloader = data_test,
        earlystopper = earlystopper
    )
      
    # predict on full test data
    output: torch.Tensor = model(data_test.x_t)
    # we want to use the EXACT same method for evaluation as in the HAR, some numerical differences where found between torch and numpy
    output_numpy: np.ndarray = output.detach().numpy().reshape(-1,)
    true_numpy: np.ndarray = data_test.y_t.detach().numpy().reshape(-1,)
    resid = output_numpy - true_numpy
    loss = np.average(resid**2)
    
    # perform out of sample prediction
    if return_prediction:
        return loss, output
    # else just return regular loss
    return loss


def fit_and_evaluateHAR(
    model: OLS, 
    data_train: DataSetNump, 
    data_test: DataSetNump, 
    normalize_features: bool = False,
    ):
    """ 
    this functions estimates the HAR model by simple OLS regression. Loss on testset is returned among the predictions itself.
    """
    if normalize_features:
        scaler = StandardScaler()
        data_train.x = scaler.fit_transform(data_train.x)
        data_test.x = scaler.transform(data_test.x)
        
    model.__init__(endog=data_train.y, exog=data_train.x)
    
    res = model.fit()
    output = res.predict(data_test.x)
    resid = data_test.y.reshape(-1,) - output.reshape(-1,)
    loss = np.average(resid**2)
    return loss, output