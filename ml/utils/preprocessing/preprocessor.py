import torch
import torch.utils.data
   
def PreProcessingData(features: torch.Tensor, targets: torch.Tensor) -> torch.utils.data.DataLoader:


    data = [(feature, target) for feature, target in zip(features, targets)]
    dataloader = torch.utils.data.DataLoader(data, batch_size)
    return dataloader