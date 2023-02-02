import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
    
def pre_process_all_data(data, train_size = None, test_size = None):
    """ 
    This function computes for the HAR and NN the features and targets, splits them into training and testing.
    The feature functions for HAR and NN are quite alike, however if we want to give the neural network more lagged features for example, it is useful to have 2 seperate HAR and NN functions.
    """
    
    # check if either train size or test size is given
    if not (train_size or test_size):
        raise Exception("Either train_size or test_size must be passed, none or both are not allowed")
    
    def compute_targets(data):
        """ 
        This just returns the unchanged data, i.e. daily realized volatility.
        Can be changed to compute the 5 or 10 day volatility for example.
        """
        return data.copy()
    
    def compute_features_har(data):
        """ Compute previous daily, weekly and monthly realized volatility """
        import statsmodels.api as sm
        # the features are computed from the targets
        targets = compute_targets(data)
        
        features = np.zeros(shape=(len(data), 3))
        features[:,0] = targets.shift(1)
        features[:,1] = targets.rolling(5).mean().shift(1)
        features[:,2] = targets.rolling(21).mean().shift(1)
        
        # for the HAR (OLS) estimation, a column of ones is added to fit a constant
        features = sm.add_constant(features)
        return features
    
    def compute_features_nn(data):
        """ Compute features for the neural network"""
        # the features are computed from the targets
        targets = compute_targets(data)
        
        features = np.zeros(shape=(len(data), 3))
        features[:,0] = targets.shift(1)
        features[:,1] = targets.rolling(5).mean().shift(1)
        features[:,2] = targets.rolling(21).mean().shift(1)
        return features
    
    # for both HAR and NN, compute the features/targets, split and return into dataset instances
    data_nn_train, data_nn_val = pre_process_data_nn(data.copy(), feature_func = compute_features_nn, target_func = compute_targets, train_size=train_size, test_size=test_size)
    data_har_train, data_har_val = pre_process_data_har(data.copy(), feature_func = compute_features_har, target_func = compute_targets, train_size=train_size, test_size=test_size)
    
    return data_nn_train, data_nn_val, data_har_train, data_har_val
    
def pre_process_data_har(data, feature_func, target_func, train_size, test_size):
    """ Computes the features (prev daily/weekly/monthly volatility) and targets (daily/weekly volatility) and returns them as train/validate datasets""" 
    # process features and targets
    targets = target_func(data).values.reshape(-1,1)
    features = feature_func(data)
    features, targets = features[22:], targets[22:]
    
    # to DataSet
    return split_and_to_dataset(features, targets, train_size=train_size, test_size=test_size, to_torch = False)
    
def pre_process_data_nn(data, feature_func, target_func, train_size, test_size):
    
    # process features and targets
    targets = target_func(data).values.reshape(-1,1)
    features = feature_func(data)
    features, targets = features[22:], targets[22:]
    
    # to DataSet
    return split_and_to_dataset(features, targets, train_size=train_size, test_size=test_size, to_torch = True)

def split_and_to_dataset(features, targets, train_size = None, test_size = None, to_torch = False):
    """ splits features and targets into training and testing sets, and loads them into DataSet class instances"""
    # split with sklearn
    features_train, features_val, targets_train, targets_val = train_test_split(features, targets, train_size = train_size, test_size=test_size, shuffle=False)
    
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
        self.x = x.astype(dtype=np.float32)
        self.y = y.astype(dtype=np.float32).reshape(-1)
        
    def __len__(self,):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
    def split(self, train_index: list, test_index: list):
        """ Build-in function to split data on given training and testing indices """
        data_train = DataSetNump(self.x[train_index], self.y[train_index])
        data_test = DataSetNump(self.x[test_index], self.y[test_index])
        return data_train, data_test
    
class DataSet(Dataset):
    """ Loads the x,y data into a Dataset instance as torch tensors """
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x_t = torch.tensor(x, dtype=torch.float32)
        self.y_t = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self,):
        return len(self.y_t)
    
    def __getitem__(self, idx: int):
        return self.x_t[idx], self.y_t[idx]
    
    def split(self, train_index: list, test_index: list):
        """ Build-in function to split data on given training and testing indices """
        data_train = DataSet(self.x_t[train_index], self.y_t[train_index])
        data_test = DataSet(self.x_t[test_index], self.y_t[test_index])
        return data_train, data_test