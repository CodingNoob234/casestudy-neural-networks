import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
    
class ForwardNeuralNetwork(nn.Module):
    """ The feed-forward neural network, with variable layers and nodes, depending on the given parameters at initialization """
    def __init__(self, 
        input_size: int, 
        output_size: int, 
        hidden_layers: list = [], 
        activation_function = nn.Sigmoid,
        seed: int = None,
        ):
        # if we want to be able to replicate the initialized parameters
        if seed:
            torch.manual_seed(seed)
        super().__init__()
        
        # the total number of layers
        layers_nodes = [input_size] + hidden_layers + [output_size]
        
        # initialize the layers
        layers = []
        for i in range(len(layers_nodes[:-2])):
            
            # create layer, normalise weights and add to list
            layer = nn.Linear(layers_nodes[i], layers_nodes[i+1], bias = True)
            self.normalize_layer(layer)
            
            # layer with bias + activation
            layers.append(layer)
            layers.append(activation_function())
            
        # the last layer is our output layer, which should not have an activation function applied to at the end
        layer = nn.Linear(layers_nodes[-2], layers_nodes[-1], bias = True)
        self.normalize_layer(layer)
        layers.append(layer)
        
        # 'link' all the functions and layers we have gathered above in a list
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        """ this function is used to process the input to output """
        return self.layers(x)
    
    def normalize_layer(self, layer: nn.Linear):
        """ normalize the weights and possible bias of a nn.Linear instance """
        # the computation for std is from literature
        std = np.sqrt(2/layer.in_features)
        
        # normalize weights
        nn.init.normal_(layer.weight, mean = 0, std = std)
        # bias is not always present
        try: nn.init.normal_(layer.bias, mean=0, std=std)
        except: pass