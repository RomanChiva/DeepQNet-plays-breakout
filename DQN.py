import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class DeepQNet(nn.Module):

    # Create the Q-Function we will be refining
    # Input(forward): Observation [86x86]
    # Output: Values for actions [4]


    def __init__(self,input_shape,hidden,output_shape) -> None:
        super().__init__()

        # Create the layers
        self.fc1 = nn.Linear(input_shape, hidden)
        self.fc2 = nn.Linear(hidden,hidden)
        self.output = nn.Linear(hidden,output_shape)
        

    def forward(self, x):

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.output(x)

        return x