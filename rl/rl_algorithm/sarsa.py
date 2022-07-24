from typing import List, Tuple
from rl import Transition, epsilon_greedy
import numpy as np

class Sarsa:
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
        # Initialize action value q(s,a) for all state-action pairs (arbitrarily)
        self.Q: np.ndarray = np.zeros(shape=shape) #np.random.normal(0, 1, size=shape)
        # Q(terminal, :) = 0
        for terminal in terminal_states:
            self.Q[terminal] = 0
    
    
    def update(self, transition: Transition) -> int:  
        # get next aciton from current policy
        next_action = self.get_action(transition.next_state)
        # compute td error
        td_error = (
            transition.reward + 
            self.gamma * self.Q[transition.next_state][next_action] - 
            self.Q[transition.current_state][transition.current_action]
        )
        # update q-value
        self.Q[transition.current_state][transition.current_action] += self.alpha * td_error
        
        # return next action
        return next_action
        
        
    def get_action(self, state: Tuple) -> int:
        return epsilon_greedy(self.Q, state, self.action_count, self.epsilon)
