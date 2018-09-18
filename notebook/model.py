import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, num_units, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        for idx, units in enumerate(num_units):
            if idx == 0:
                modules = [nn.Linear(state_size, units)]
            else:
                modules.append(nn.Linear(num_units[idx-1], units))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(num_units[-1], action_size))
        self.model = nn.Sequential(*modules)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        return self.model(state)
