import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class DeepQNet(nn.Module):

    # Create the Q-Function we will be refining
    # Input(forward): Observation [86x86]
    # Output: Values for actions [4]


    def __init__(self,input_shape,output_shape) -> None:
        super().__init__()

        # Create the layers
        self.conv1 = nn.Conv2d(in_channels=input_shape,out_channels=16,kernel_size=8,stride=4)
        self.conv2 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=4,stride=2)
        self.fc = nn.Linear(2592,256)
        self.output = nn.Linear(256,output_shape)
        

    def forward(self, x):

        # Convolutions and relus
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        
        # Flatten
        x = torch.flatten(x)

        #Linear layers and output
        x = self.fc(x)
        x = F.relu(x)
        x = self.output(x)

        return x