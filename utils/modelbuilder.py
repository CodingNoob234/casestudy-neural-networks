import torch
import torch.nn.functional as F
import torch.nn as nn    
    
class ForwardNeuralNetwork(nn.Module): 
    """ 
    A neural network
    This class is build in such a manner, that depending on the length of the hidden_layers parameter, the number of hidden layers is changed to that.
    Usually this is done by hand, but through this way, makes it very flexible to adjust, also for cross validating multiple model specifications.
    """
    def __init__(self, input_size: int, output_size: int, hidden_layers: list = []):
        torch.manual_seed(3407)
        super().__init__()
        
        layers = [input_size] + hidden_layers + [output_size]
        self.n_layers = len(layers)-1
        
        # add layers to initialize models
        init_layers = []
        for i, layer in enumerate(layers[:-1], 1):
            init_layer = nn.Linear(layers[i-1], layers[i])
            
            # equal to: self.fc{i} = nn.Linear(layers[i-1], layer[i])
            self.__setattr__(f"fc{i}", init_layer)
            init_layers.append(init_layer)
            
        self.__setattr__(f"fc{i}", nn.Linear(layers[i-1], layers[i])   )
        
        # store the layers as a list as well
        self.__setattr__("ls", init_layers)
        
    def forward(self, x: torch.Tensor):
        for i in range(self.n_layers-1):
            layer = self.__getattr__(f"fc{i+1}")
            x = F.relu(layer(x))
        layer = self.__getattr__(f"fc{self.n_layers}")
        x = layer(x)
        return x