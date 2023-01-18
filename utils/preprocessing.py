import numpy as np
import torch

from torch.utils.data import Dataset

from utils.functions import get_ticker_daily_close

class PreProcessor():
    def __init__(self, ):
        pass
    
def pre_process_data_har(stock: str = "KO"):
    data = get_ticker_daily_close(stock)
    d = DataSet()
    return d

def pre_process_data_nn(stock: str = "KO"):
    d = DataSet()
    return d

class DataSetNump(Dataset):
    """ Load the x,y data in a Dataset instance as numpy arrays """
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y
        
    def __len__(self,):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
class DataSet(Dataset):
    """ Loads the x,y data into a Dataset instance as torch tensors """
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x_t = torch.tensor(x, dtype=torch.float32)
        self.y_t = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self, ):
        return len(self.y_t)
    
    def __getitem__(self, idx):
        return self.x_t[idx], self.y_t[idx]