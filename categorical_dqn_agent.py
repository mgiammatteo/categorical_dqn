import numpy as np
import random
from collections import namedtuple, deque
import torch.autograd as autograd 

from model import CategorcialDQN

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network
V_min = -10
V_max = 10
N = 51                  # we want to implement C51 - distributional RL
delta_z = (V_max-V_min)/(N-1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, num_atoms, V_min, V_max):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.num_atoms = num_atoms
        self.V_min = V_min
        self.V_max = V_max

        # Q-Network
        self.qnetwork_local = CategorcialDQN(state_size, action_size, seed, self.num_atoms, self.V_min, self.V_max).to(device)
        self.qnetwork_target = CategorcialDQN(state_size, action_size, seed, self.num_atoms, self.V_min, self.V_max).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def projection_distribution(self, next_states, rewards, dones):
        batch_size  = next_states.size(0)
      
        rewards = rewards.data.cpu()
        dones = dones.data.cpu()
        
        delta_z = float(self.V_max - self.V_min) / (self.num_atoms - 1)
        support = torch.linspace(self.V_min, self.V_max, self.num_atoms)
        
        next_dist   = self.qnetwork_target(next_states).data.cpu() * support
      
        next_action = next_dist.sum(2)
       
        # This is equivalent to argMax as the max operation returns the max of the four actions and their index!
        next_action = next_action.max(1)[1]
        
        next_action = next_action.unsqueeze(1).unsqueeze(1).expand(next_dist.size(0), 1, next_dist.size(2))
        
        next_dist   = next_dist.gather(1, next_action).squeeze(1)
        
        rewards = rewards.expand_as(next_dist)
        dones   = dones.expand_as(next_dist)
        support = support.unsqueeze(0).expand_as(next_dist)

        Tz = rewards + (1 - dones) * GAMMA * support
        Tz = Tz.clamp(min=self.V_min, max=self.V_max)
        b  = (Tz - self.V_min) / delta_z
        l  = b.floor().long()
        u  = b.ceil().long()

        offset = torch.linspace(0, (batch_size - 1) * self.num_atoms, batch_size).long()\
                        .unsqueeze(1).expand(batch_size, self.num_atoms)

        proj_dist = torch.zeros(next_dist.size())    
        proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
        proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))

        return proj_dist

    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            # In Categorical DQN the NN outputs a set on N probabilities for each action
            action_values = self.qnetwork_local(state)
            # this is a matrix of |A| x N, where |A| is the size of the action space and N is the 
            # number of paramters of our value distribution parameterisation
            M = action_values.data.cpu().numpy()
            Q = M * torch.linspace(self.V_min, self.V_max, self.num_atoms).view(1, 1, self.num_atoms).to(device)
            Q = Q.sum(dim=2).max(1)[1].view(1, 1)
        self.qnetwork_local.train()
  
        # Epsilon-greedy action selection
        if random.random() > eps:
            return Q.item()
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        
        proj_dists = self.projection_distribution(next_states, rewards, dones)
        proj_dists = proj_dists.float().to(device)
       
        dists = self.qnetwork_local(states)
        actions = actions.unsqueeze(1).expand(BATCH_SIZE, 1, self.num_atoms)
        dists = dists.gather(1, actions).squeeze(1)
        dists.data.clamp_(0.01, 0.99)
        
        loss = - (proj_dists * dists.log()).sum(-1).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)