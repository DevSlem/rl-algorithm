from typing import Tuple
from dataclasses import dataclass
import numpy as np
import random

@dataclass(frozen=True)
class Transition:
    current_state: Tuple
    current_action: int
    next_state: Tuple
    reward: float

class Sarsa:
    def __init__(self, state_shape, action_count, terminal_states) -> None:
        self.action_count = action_count
        self.epsilon = 0.1
        
        shape = state_shape + (action_count,)
        # Initialize action value q(s,a) for all state-action pairs (arbitrarily)
        self.Q = np.random.normal(0, 1, size=shape)
        # Q(terminal, :) = 0
        for terminal in terminal_states:
            self.Q[terminal] = 0
    
    def update(self, transition: Transition, alpha = 0.1, gamma = 0.1):
        # Compute td error
        next_action = self.get_action(transition.next_state)
        
        td_error = (
            transition.reward + 
            gamma * self.Q[transition.next_state][next_action] - 
            self.Q[transition.current_state][transition.current_action]
        )

        # Update action value
        self.Q[transition.current_state][transition.current_action] += alpha * td_error
        
    def get_action(self, state: Tuple) -> int:
        p = random.random()
        if p > self.epsilon:
            return np.argmax(self.Q[state])
        else:
            return random.randint(0, self.action_count - 1)