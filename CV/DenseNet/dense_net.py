import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, List
from math import floor, ceil

def connection(in_channels : int, out_channels : int, kernel_size : int):
    padding = (floor((kernel_size - 1)/2), ceil((kernel_size - 1)/2))
    return nn.Sequential(nn.ReLU(),
                         nn.BatchNorm2d(out_channels),
                         nn.Conv2d(in_channels, 
                                   out_channels, 
                                   kernel_size, 
                                   stride=1, 
                                   padding=padding))
    
def transition(in_channels : int,
               growth_rate : int, 
               num_layers : int, 
               compression : float,
               in_size : int,
               out_size : int):
    pool_factor = floor(in_size / out_size)
    out_channels = int(compression * (in_channels + num_layers * growth_rate))
    return nn.Sequential(nn.BatchNorm2d(in_channels + num_layers * growth_rate),
                         nn.Conv2d(in_channels=in_channels + num_layers * growth_rate,
                                   out_channels=out_channels,
                                   kernel_size=in_size - out_size * pool_factor + 1,
                                   stride=1),
                         nn.AvgPool2d(kernel_size=pool_factor, stride=pool_factor))

class Block(nn.Module):

    def __init__(self, 
                 num_layers : int, 
                 in_channels : int, 
                 growth_rate : int, 
                 kernel_size : int,
                 in_size : int,
                 out_size : int,
                 bottleneck : bool = True,
                 bottleneck_in : int = 4,
                 compression : float = 1.0):
        self.conns = []
        for i in range(num_layers):
            if bottleneck:
                layers = nn.Sequential(connection(in_channels=in_channels + i * growth_rate, 
                                                  out_channels=bottleneck_in * growth_rate, 
                                                  kernel_size=1),
                                       connection(in_channels=bottleneck_in * growth_rate,
                                                  out_channels=growth_rate,
                                                  kernel_size=kernel_size))
                self.conns.append(layers)
            else:
                self.conns.append(connection(in_channels=in_channels + i * growth_rate,
                                             out_channels=growth_rate,
                                             kernel_size=1))
        self.transition = transition(in_channels, growth_rate, num_layers, compression, in_size, out_size)
        
    def __call__(self, x):
        state = x
        for connection in self.conns:
            y = connection(state)
            state = torch.cat([state, y], axis=2)
        return self.transition(state)
    

class DenseNet():
    
    def __init__(self, 
                 growth_rate : int, 
                 input_size : int,
                 hidden_sizes : Sequence[int],
                 num_layers : int,
                 bottleneck : bool = True,
                 bottleneck_in : int = 4,
                 compression : float = 1.0):
        sizes = [input_size] + list(hidden_sizes)
        self.blocks = []
        for in_size, out_size in zip(sizes[:-1], sizes[1:]):
            self.blocks.append(Block(num_layers,
                                     ))
            
        
    
    def __call__(self, x):
        pass    
    
    def loss(self, x):
        return F.nll_loss(x, self(x))
        

