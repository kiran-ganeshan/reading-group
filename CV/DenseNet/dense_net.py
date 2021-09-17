import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, Tuple
from collections import OrderedDict
from math import floor, ceil

def connection(in_channels : int, out_channels : int, kernel_size : int):
    padding = (floor((kernel_size - 1)/2), ceil((kernel_size - 1)/2))
    return nn.Sequential(
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(in_channels, 
                  out_channels, 
                  kernel_size, 
                  stride=1, 
                  padding=padding)
        )
    
def transition(in_channels : int,
               compression : float,
               pool : bool = True):
    return nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.Conv2d(in_channels=in_channels,
                  out_channels=int(compression * in_channels),
                  kernel_size=1,
                  stride=1),
        nn.AvgPool2d(kernel_size=2, stride=2) if pool else nn.Identity()
    )

class Block(nn.Module):

    def __init__(self, 
                 num_layers : int, 
                 in_channels : int, 
                 growth_rate : int, 
                 kernel_size : int,
                 bottleneck : bool = True,
                 bn_size : int = 4,
                 dropout : float = 0.0,
                 use_transition : bool = True):
        self.conns = nn.ModuleList()
        for i in range(num_layers):
            out_channels = (bn_size if bottleneck else 1) * growth_rate,
            self.conns.append(connection(in_channels=in_channels + i * growth_rate, 
                                         out_channels=out_channels,
                                         kernel_size=1))
            if bottleneck:
                self.conns.append(connection(in_channels=bn_size * growth_rate,
                                             out_channels=growth_rate,
                                             kernel_size=kernel_size))
        self.dropout = nn.Dropout2d(dropout)
        
    def forward(self, x):
        state = x
        for connection in self.conns:
            y = connection(state)
            state = torch.cat([state, y], axis=1)
        state = self.dropout(state)
        return state
    

class DenseNet():

    def __init__(self, 
                 growth_rate : int, 
                 block_depths : Tuple[int, int, int, int],
                 num_init_features : int,
                 num_classes : int,
                 bottleneck : bool = True,
                 bn_size : int = 4,
                 dropout : float = 0,
                 compression : float = 1.0):
        super(DenseNet, self).__init__()
        
        # Initial convolution, norm, relu, and pool
        self.init_block = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))
        
        # Dense blocks
        self.blocks = nn.ModuleList()
        num_features = num_init_features
        for i, num_layers in enumerate(block_depths):
            self.blocks.append(Block(num_layers=num_layers,
                                     in_channels=num_features,
                                     growth_rate=growth_rate,
                                     kernel_size=3,
                                     bottleneck=bottleneck,
                                     bn_size=bn_size,
                                     compression=compression,
                                     dropout=dropout))
            num_features += num_layers * growth_rate
            if i != num_layers - 1:
                self.blocks.append(transition(in_channels=num_features, 
                                              compression=compression))
                num_features = int(compression * num_features)
        
        # Final norm and classifier
        self.norm = nn.BatchNorm2d(num_features)
        self.classifier = nn.Sequential(nn.Linear(num_features, num_classes),
                                        nn.LogSoftmax())
        
        # Initialization scheme
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
    
    def __call__(self, x):
        pass    
    
    def loss(self, x):
        return F.nll_loss(x, self(x))
        

