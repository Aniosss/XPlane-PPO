import torch
from torch import nn
import numpy as np


class NeuralNetwork(nn.Module):
    """
        Standard input_dim-128-64-output_dim NN
    """
    def __init__(self, input_dim, output_dim):
        super(NeuralNetwork, self).__init__()

        self.l1 = nn.Linear(input_dim, 128)
        self.l2 = nn.Linear(128, 64)
        self.l3 = nn.Linear(64, output_dim)

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)

        act1 = nn.ELU()(self.l1(obs))
        act2 = nn.ELU()(self.l2(act1))
        actions = torch.tanh(self.l3(act2))

        return actions
