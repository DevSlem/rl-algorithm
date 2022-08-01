from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from rl import Transition, Agent, epsilon_greedy_dnn, OnPolicyReplay
from rl.util import Decay, NoDecay

class DeepSarsa(Agent):
    def __init__(self,
                 q_value_net: nn.Module,
                 optimizer: optim.Optimizer,
                 action_count: int,
                 loss_func = nn.MSELoss(),
                 onpolicy_replay: OnPolicyReplay = OnPolicyReplay(32),
                 epsilon_decay: Decay = NoDecay(0.1),
                 gamma = 0.99,
                 device: torch.device = None) -> None:
        self.q_value_net = q_value_net
        self.optimizer = optimizer
        self.action_count = action_count
        self.loss_func = loss_func
        self.onpolicy_replay = onpolicy_replay
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.epsilon = epsilon_decay.step()
        self.step = 0
        self.device = device
        
    def update(self, transition: Transition) -> Any:
        # if reached training frequency
        if self.onpolicy_replay.is_full:
            # sample from the onpolicy replay
            prev_transitions = self.onpolicy_replay.sample()
            # for the batch learning
            current_states, current_actions, next_states, rewards, terminated_arr = (
                Transition.to_tensor_batch(prev_transitions, self.device)
            )
            # get a next action from the next transition
            next_actions = torch.cat(
                [current_actions[1:], 
                 transition.to_tensor(self.device).current_action.unsqueeze(0)]
            )
            # compute td loss
            loss = self.compute_td_loss(
                current_states,
                current_actions,
                next_states,
                next_actions,
                rewards,
                terminated_arr
            )
            # backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.step += 1
            # set epsilon
            self.epsilon = self.epsilon_decay.step()
            
        # add new transition
        self.onpolicy_replay.add(transition)
            
    def get_action(self, state: torch.Tensor) -> Any:
        return epsilon_greedy_dnn(
            self.q_value_net,
            torch.FloatTensor(state, device=self.device),
            self.action_count,
            self.epsilon
        )
            
    def compute_td_loss(self, 
                        current_states: torch.Tensor, 
                        current_actions: torch.Tensor, 
                        next_states: torch.Tensor, 
                        next_actions: torch.Tensor, 
                        rewards: torch.Tensor, 
                        terminated_arr: torch.Tensor):
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
        # compute td loss
        td_target = rewards + self.gamma * (1 - terminated_arr) * next_q_value
        td_loss = self.loss_func(q_value, td_target)
        
        return td_loss