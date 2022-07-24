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
                 device: torch.device = None) -> None:
        
        self.policy_net = policy_net
        self.optimizer = optimizer
        self.dist_generator = dist_generator
        self.device = device
        self.gamma = gamma
        self.current_action_log_prob = None
        self.action_log_probs = []
        self.rewards = []
        self.__loss = 0.0
        
    def reset(self) -> None:
        return super().reset()
        
    def start_episode(self):
        self.current_action_log_prob = None
        self.action_log_probs = []
        self.rewards = []
        self.__loss = 0.0
        
    def update(self, transition: rl.Transition):
        if self.current_action_log_prob is None:
            raise Exception("You need to call get_action() method before call it.")
        # collect data
        self.rewards.append(transition.reward)
        self.action_log_probs.append(self.current_action_log_prob)
        self.current_action_log_prob = None
        
    def end_episode(self):
        # terminal time step
        T = len(self.rewards)
        returns = np.empty(T, dtype=np.float32)
        current_return = 0.0
        # compute returns for all time steps in the episode
        for t in reversed(range(T)):
            current_return = self.rewards[t] + self.gamma * current_return
            returns[t] = current_return
        # convert to tensor
        returns = torch.tensor(returns).to(device=self.device)
        log_probs = torch.stack(self.action_log_probs).to(device=self.device)
        # compute baseline
        baseline = returns.mean()
        # compute loss
        loss = -(returns - baseline) * log_probs
        loss = torch.sum(loss)
        # back propagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.__loss = loss.cpu().detach().item()
        
    @property
    def loss(self) -> np.ndarray:
        return self.__loss
    
    def get_action(self, state: np.ndarray):
        x = torch.from_numpy(state).to(device=self.device, dtype=torch.float32)
        # foward propagation
        pdparam = self.policy_net(x)
        # generate a policy distribution
        pd = self.dist_generator(pdparam)
        # sample an action from the policy distribution
        action = pd.sample()
        # store for training
        self.current_action_log_prob = pd.log_prob(action)
        return action.cpu().detach().item()
