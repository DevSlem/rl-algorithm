from typing import Any, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from rl import Transition, Agent, epsilon_greedy_dnn

class DeepSarsa(Agent):
    def __init__(self,
                 q_value_net: nn.Module,
                 optimizer: optim.Optimizer,
                 action_count: int,
                 loss_func = None,
                 gamma = 0.99,
                 epsilon = 0.1,
                 batch_size = 32,
                 device: torch.device = None) -> None:
        self.q_value_net = q_value_net
        self.optimizer = optimizer
        self.action_count = action_count
        self.loss_func = nn.MSELoss if loss_func is None else loss_func
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.device = device
        
        self.transitions = []

    def update(self, transition: Transition) -> Any:
        self.transitions.append(transition)
        if len(self.transitions) == self.batch_size + 1:
            loss = self.compute_td_error()
            # backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # remove all transitions except for the last one
            temp = self.transitions[-1]
            self.transitions.clear()
            self.transitions.append(temp)
            
    def get_action(self, state) -> Any:
        return epsilon_greedy_dnn(
            self.q_value_net,
            torch.FloatTensor(state, device=self.device),
            self.action_count,
            self.epsilon
        )
            
    def compute_td_error(self):
        # for the batch learning
        current_states, current_actions, next_states, rewards, terminated_arr = (
            Transition.to_tensor_batch(self.transitions[:-1], self.device)
        )
        # get a next action from the next transition
        next_actions = torch.cat(
            [current_actions[1:], self.transitions[-1].to_tensor(self.device)]
        )
        # estimate action values for the current state
        q_values = self.q_value_net(current_states)
        # estimate action values for the next state to compute a target of sarsa
        with torch.no_grad():
            next_q_values = self.q_value_net(next_states)
        # value of the selected current action
        q_value = torch.gather(
            q_values,
            -1,
            current_actions.long().unsqueeze(-1)
        ).squeeze(-1)
        # value of the selected next action
        next_q_value = torch.gather(
            next_q_values,
            -1,
            next_actions.long().unsqueeze(-1)
        ).squeeze(-1)
        # compute td error
        td_target = rewards + self.gamma * (1 - terminated_arr) * next_q_value
        td_error = self.loss_func(q_value, td_target)
        
        return td_error