import numpy as np
import torch

class PreProcessor():
    def __init__(self, ):
        pass
    
    
class DataSet(Dataset):
    """ Loads the x,y data into a Dataset instance """
    def __init__(self, x, y):
        self.x_t = torch.Tensor(x, dtype=torch.float32)
        self.y_t = torch.Tensor(y, dtype=torch.float32)
        
    def __len__(self, ):
        return len(self.y_t)
    
    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]
    

def data_to_loaders(features_train_tensor: torch.Tensor, features_test_tensor: torch.Tensor, targets_train_tensor: torch.Tensor, targets_test_tensor: torch.Tensor) -> tuple:
    """ give training and testing features/targets as torch.Tensors, and return train/testloader"""

    # to data loader so torch can handle the data efficiently
    trainloader = torch.utils.data.DataLoader( 
        [(feature, target) for feature, target in zip(features_train_tensor, targets_train_tensor)], 
        batch_size = 20)
    
    testloader = torch.utils.data.DataLoader(
        [(feature, target) for feature, target in zip(features_test_tensor, targets_test_tensor)]
    )
    
    return trainloader, testloader