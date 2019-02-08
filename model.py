import torch
import torch.nn as nn
import torch.nn.functional as F

class CategorcialDQN(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, num_atoms, Vmin, Vmax, fc1_units=256, fc2_units=256):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(CategorcialDQN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.num_atoms    = num_atoms
        self.Vmin         = Vmin
        self.Vmax         = Vmax
        self.num_actions = action_size
        
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size * self.num_atoms)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x.view(-1, self.num_actions, self.num_atoms), dim=2)
