import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence

def connection(in_channels : int, out_channels : int, kernel_size : int):
    return nn.Sequential(nn.ReLU(),
                         nn.BatchNorm2d(out_channels),
                         nn.Conv2d(in_channels, 
                                   out_channels, 
                                   kernel_size, 
                                   stride=1, 
                                   padding=kernel_size - 1))
    
def transition(in_channels : int, growth_rate : int, num_layers : int, compression : float):
    out_channels = int(compression * (in_channels + num_layers * growth_rate))
    return nn.Sequential(nn.BatchNorm2d(in_channels + num_layers * growth_rate),
                         nn.Conv2d(in_channels=in_channels + num_layers * growth_rate,
                                   out_channels=out_channels),
                         nn.AvgPool2d(kernel_size=2, stride=2))

class Block(nn.Module):

    def __init__(self, 
                 num_layers : int, 
                 in_channels : int, 
                 growth_rate : int, 
                 out_channels : int, 
                 kernel_size : int,
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
        self.transition = transition(in_channels, growth_rate, num_layers, compression)
        
    def __call__(self, x):
        state = x
        for connection in self.conns:
            y = connection(state)
            state = torch.cat([state, y], axis=2)
        return self.transition(state)

class DenseNet():
    
    def __init__(self, 
                 growth_rate : int, 
                 image_sizes : Sequence[int],
                 bottleneck : bool = True,
                 bottleneck_in : int = 4,
                 compression : float = 1.0):
        
    
    def __call__(self, x):
        pass    
    
    def loss(self, x):
        return F.nll_loss(x, self(x))
        

