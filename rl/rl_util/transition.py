from __future__ import annotations
from typing import Any, List, NamedTuple
import numpy as np
import torch

class Transition(NamedTuple):
    current_state: Any
    current_action: Any
    next_state: Any
    reward: float
    terminated: bool
    
    def to_tabular(self):
        transition = Transition(
            tuple(self.current_state),
            int(self.current_action),
            tuple(self.next_state),
            self.reward,
            self.terminated
        )
        return transition
    
    def to_numpy(self):
        transition = Transition(
            np.array(self.current_state),
            np.array(self.current_action),
            np.array(self.next_state),
            self.reward,
            self.terminated
        )
        return transition
    
    def to_tensor(self, device: torch.device = None, requires_grad: bool = False):
        transition = Transition(
            torch.tensor(self.current_state, device=device, requires_grad=requires_grad),
            torch.tensor(self.current_action, device=device, requires_grad=requires_grad),
            torch.tensor(self.next_state, device=device, requires_grad=requires_grad),
            self.reward,
            self.terminated
        )
        return transition
    
    @staticmethod
    def to_tensor_batch(transitions: list[Transition], device: torch.device = None, requires_grad: bool = False):
        current_states = []
        current_actions = []
        next_states = []
        rewards = []
        terminated_arr = []
        
        for transition in transitions:
            current_states.append(transition.current_state)
            current_actions.append(transition.current_action)
            next_states.append(transition.next_state)
            rewards.append(transition.reward)
            terminated_arr.append(transition.terminated)
            
        current_states = torch.tensor(np.array(current_states), device=device, requires_grad=requires_grad)
        current_actions = torch.tensor(np.array(current_actions), device=device, requires_grad=requires_grad)
        next_states = torch.tensor(np.array(next_states), device=device, requires_grad=requires_grad)
        rewards = torch.tensor(rewards, device=device, requires_grad=requires_grad)
        terminated_arr = torch.tensor(terminated_arr, device=device, requires_grad=requires_grad).int()
        
        return current_states, current_actions, next_states, rewards, terminated_arr