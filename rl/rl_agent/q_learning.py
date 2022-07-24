from typing import List, Tuple
from rl import TabularQAgent, Transition, epsilon_greedy
import numpy as np


class QLearning(TabularQAgent):
    def __init__(self, obs_shape: tuple, 
                 action_count: int, 
                 terminal_states: List[Tuple] = None, 
                 epsilon = 0.1,
                 alpha = 0.1,
                 gamma = 0.9) -> None:
        super().__init__(obs_shape, action_count, terminal_states)
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
            
    def update(self, transition: Transition) -> None:
        Q = self._Q
        transition = transition.to_tabular()
        
        # get maximum q-value in next state
        target_q = np.max(Q[transition.next_state]) 
        # compute td error
        td_error = transition.reward + self.gamma * target_q - Q[transition.current_state][transition.current_action]
        # update q-value
        Q[transition.current_state][transition.current_action] += self.alpha * td_error
            
    def get_action(self, state) -> int:
        return epsilon_greedy(self._Q, tuple(state), self.action_count, self.epsilon)
