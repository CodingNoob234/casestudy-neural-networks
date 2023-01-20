import torch
import torch.nn.functional as F
import torch.nn as nn    
    
class ForwardNeuralNetwork(nn.Module):
    """ The feed-forward neural network, with variable layers and nodes, depending on the given parameters at initialization """
    def __init__(self, 
        input_size: int, 
        output_size: int, 
        hidden_layers: list = [], 
        activation_function = nn.Sigmoid,
        seed: int = None,
        ):
        # if we want to replicate the initialization
        if seed:
            torch.manual_seed(seed)
        super().__init__()
        
        # the total number of layers
        layers_nodes = [input_size] + hidden_layers + [output_size]
        
        # initialize the layers
        layers = []
        for i in range(len(layers_nodes[:-2])):
            layers.append(nn.Linear(layers_nodes[i], layers_nodes[i+1]))
            layers.append(activation_function())
            
        # the last layer is our output layer, which should not have an activation function applied to
        layers.append(nn.Linear(layers_nodes[-2], layers_nodes[-1]))
        
        # this 'links' all the functions and layers we have gathered above in a list
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        """ this function is used to process the input to output """
        return self.layers(x)