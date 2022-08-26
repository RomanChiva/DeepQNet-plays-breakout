import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class DeepQNet(nn.Module):

    # Create the Q-Function we will be refining
    # Input(forward): Observation [86x86]
    # Output: Values for actions [4]


    def __init__(self,output_shape) -> None:
        super().__init__()

        # Create the layers

        self.conv1 = nn.Conv2d(in_channels=4,out_channels=32,kernel_size=8,stride=4)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1)
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=2,stride=2)
        self.fully_connected = nn.Linear(2816,512)
        self.output = nn.Linear(512,output_shape)

    
        

    def forward(self, x):

        # Convolutions and relus
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        
        # Flatten
        x = torch.flatten(x, start_dim=1) # Flatten the samples but not the batch!!!

        #Linear layers and output
        x = self.fully_connected(x)
        x = F.relu(x)
        x = self.output(x)

        return x



class DeepQNet2(nn.Module):

    # Create the Q-Function we will be refining
    # Input(forward): Observation [86x86]
    # Output: Values for actions [4]


    def __init__(self,output_shape) -> None:
        super().__init__()

        # Create the layers

        self.conv1 = nn.Conv2d(in_channels=4,out_channels=8,kernel_size=5,stride=3)
        self.conv2 = nn.Conv2d(in_channels=8,out_channels=4,kernel_size=3,stride=2)
        self.fully_connected = nn.Linear(420,105)
        self.output = nn.Linear(105,output_shape)

    
        

    def forward(self, x):

        # Convolutions and relus
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        
        # Flatten
        x = torch.flatten(x, start_dim=1) # Flatten the samples but not the batch!!!

        #Linear layers and output
        x = self.fully_connected(x)
        x = F.relu(x)
        x = self.output(x)

        return x