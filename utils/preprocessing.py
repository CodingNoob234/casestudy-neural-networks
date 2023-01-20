import numpy as np
import torch

from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split

import statsmodels.api as sm

from utils.functions import get_ticker_daily_close
    
def pre_process_all_data(stock: str = "KO", train_size = .8):
    """ This functions computes for the HAR and NN the features and targets, splits them"""
    data = get_ticker_daily_close(stock)
    
    def compute_targets(data):
        return data.apply(np.log).diff().apply(lambda x: x**2)
    
    def compute_features_har(data):
        targets = compute_targets(data)
        features = np.zeros(shape=(len(data), 3))
        features[:,0] = targets.shift(1)
        features[:,1] = targets.rolling(5).mean().shift(1)
        features[:,2] = targets.rolling(21).mean().shift(1)
        
        # add constant for har features and drop nan values
        features = sm.add_constant(features)
        return features
    
    def compute_features_nn(data):
        targets = compute_targets(data)
        features = np.zeros(shape=(len(data), 3))
        features[:,0] = targets.shift(1)
        features[:,1] = targets.rolling(5).mean().shift(1)
        features[:,2] = targets.rolling(21).mean().shift(1)
        return features
    
    # for both HAR and NN, compute the features/targets, split and return into dataset instances
    data_nn_train, data_nn_val = pre_process_data_nn(data.copy(), feature_func = compute_features_nn, target_func = compute_targets, train_size=train_size)
    data_har_train, data_har_val = pre_process_data_har(data.copy(), feature_func = compute_features_har, target_func = compute_targets, train_size=train_size)
    
    return data_nn_train, data_nn_val, data_har_train, data_har_val
    
def pre_process_data_har(data, feature_func, target_func, train_size = .8):
    """ Computes the features (prev daily/weekly/monthly volatility) and targets (daily/weekly volatility) and returns them as train/validate datasets""" 
    # process features and targets
    targets = target_func(data).values.reshape(-1,1)
    features = feature_func(data)
    features, targets = features[22:], targets[22:]
    
    # to DataSet
    return split_and_to_dataset(features, targets, train_size=train_size, to_torch = False)
    
def pre_process_data_nn(data, feature_func, target_func, train_size = .8):
    # process features and targets
    targets = target_func(data).values.reshape(-1,1)
    features = feature_func(data)
    features, targets = features[22:], targets[22:]
    
    # to DataSet
    return split_and_to_dataset(features, targets, train_size=train_size, to_torch = True)

def split_and_to_dataset(features, targets, train_size = .8, to_torch = False):
    """ splits features and targets into training and testing sets, and loads them into DataSet class instances"""
    # split with sklearn
    features_train, features_val, targets_train, targets_val = train_test_split(features, targets, train_size = train_size, shuffle=False)
    
    # to DataSet instances
    if to_torch:
        data_train = DataSet(features_train, targets_train)
        data_val = DataSet(features_val, targets_val)
    else:
        data_train = DataSetNump(features_train, targets_train)
        data_val = DataSetNump(features_val, targets_val)
    return data_train, data_val

class DataSetNump(Dataset):
    """ Load the x,y data in a Dataset instance as numpy arrays """
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y
        
    def __len__(self,):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
    def split(self, train_index: list, test_index: list):
        data_train = DataSetNump(self.x[train_index], self.y[train_index])
        data_test = DataSetNump(self.x[test_index], self.y[test_index])
        return data_train, data_test
    
class DataSet(Dataset):
    """ Loads the x,y data into a Dataset instance as torch tensors """
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x_t = torch.tensor(x, dtype=torch.float32)
        self.y_t = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self, ):
        return len(self.y_t)
    
    def __getitem__(self, idx: int):
        return self.x_t[idx], self.y_t[idx]
    
    def split(self, train_index: list, test_index: list):
        data_train = DataSet(self.x_t[train_index], self.y_t[train_index])
        data_test = DataSet(self.x_t[test_index], self.y_t[test_index])
        return data_train, data_test