import torch
import torch.nn as nn

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, num_units, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            num_units (list): Number of hidden-nodes in each fully-connected layer.
              Note the model will have a number of fully-connected layers equal to
              the size of num_units.
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        # Add fully-connected layers.
        for idx, units in enumerate(num_units):
            if idx == 0:
                modules = [nn.Linear(state_size, units)]
            else:
                modules.append(nn.Linear(num_units[idx-1], units))
            modules.append(nn.ReLU())
        # Add final layer with output equal to action space size.
        modules.append(nn.Linear(num_units[-1], action_size))
        self.model = nn.Sequential(*modules)

    def forward(self, state):
        # Apply model to input state.
        return self.model(state)