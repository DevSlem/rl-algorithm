from typing import List, Tuple
from rl import Transition, epsilon_greedy
import numpy as np


class QLearning:
    def __init__(self, state_shape: Tuple, 
                 action_count: int, 
                 terminal_states: List[Tuple], 
                 epsilon = 0.1,
                 alpha = 0.1,
                 gamma = 0.9) -> None:
        
        self.action_count = action_count
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        
        shape = state_shape + (action_count,)
        self.Q: np.ndarray = np.zeros(shape=shape)
            
            
    def update(self, transition: Transition) -> None:       
        # get maximum q-value in next state
        target_q = np.max(self.Q[transition.next_state]) 
        # compute td error
        td_error = transition.reward + self.gamma * target_q - self.Q[transition.current_state][transition.current_action]
        # update q-value
        self.Q[transition.current_state][transition.current_action] += self.alpha * td_error
            
            
    def get_action(self, state: Tuple) -> int: 
        return epsilon_greedy(self.Q, state, self.action_count, self.epsilon)
