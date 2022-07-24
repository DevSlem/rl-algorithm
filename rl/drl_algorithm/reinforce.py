import rl
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Distribution
from torch.optim import Optimizer
from typing import Callable

class Reinforce(rl.Agent):
    def __init__(self,
                 policy_net: nn.Module,
                 optimizer: Optimizer,
                 dist_generator: Callable[[torch.Tensor], Distribution],
                 gamma = 0.99,
                 device = None) -> None:
        
        self.policy_net = policy_net
        self.optimizer = optimizer
        self.dist_generator = dist_generator
        self.device = device
        self.gamma = gamma
        self.current_action_log_prob = 0.0
        self.current_return = 0.0
        self.__loss = torch.tensor(0)
        
    def start_episode(self):
        self.current_action_log_prob = 0.0
        self.current_return = 0.0
        self.__loss = torch.tensor(0)
        
    def update(self, transition: rl.Transition):
        self.current_return = transition.reward + self.gamma * self.current_return
        self.__loss -= self.current_return * self.current_action_log_prob
        self.current_action_log_prob = 0.0
        
    def end_episode(self):
        self.optimizer.zero_grad()
        self.__loss.backward()
        self.optimizer.step()
        
    @property
    def loss(self) -> np.ndarray:
        return self.__loss.cpu().detach().numpy()
    
    def get_action(self, state):
        x = torch.from_numpy(state).to(device=self.device, dtype=torch.float32)
        # foward propagation
        pdparam = self.policy_net(x)
        # generate a policy distribution
        pd = self.dist_generator(pdparam)
        # sample an action from the policy distribution
        action = pd.sample()
        # store for training
        log_prob = pd.log_prob(action)
        self.current_action_log_prob = log_prob
        return action.item()
