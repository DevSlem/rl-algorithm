from typing import Tuple
import numpy as np
import torch
import torch.nn as nn

def epsilon_greedy(q_values, state: Tuple, action_count: int, epsilon = 0.1) -> int:
    p = np.random.rand()
    if p > epsilon:
        return np.argmax(q_values[state])
    else:
        return np.random.randint(action_count)
    
def epsilon_greedy_dnn(q_value_net: nn.Module, state: torch.Tensor, action_count: int, epsilon=0.1):
    if np.random.rand() > epsilon:
        with torch.no_grad():
            q_values = q_value_net(state)
            return torch.argmax(q_values, dim=1).cpu().detach().numpy()
    else:
        return np.random.randint(action_count, size=state.shape[0])
    
def epsilon_greedy_distribution(q_values, 
                                state: Tuple, 
                                action_count: int, 
                                epsilon = 0.1) -> np.ndarray:
    # epsilon-greedy policy probailities
    greed_action_prob = 1.0 - epsilon + epsilon / action_count
    non_greedy_action_prob = epsilon / action_count
    # get greedy action given the state
    greedy_action = np.argmax(q_values[state])
    # set policy distribution for current state
    policy_distribution = np.zeros(action_count)
    policy_distribution[greedy_action] = greed_action_prob # greedy action
    policy_distribution[np.arange(action_count) != greedy_action] = non_greedy_action_prob # non greedy action
    return policy_distribution