from typing import List, Tuple
from transition import Transition
import numpy as np
import rl_utility

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
        """ Update q-values with sarsa.

        Args:
            transition (Transition): transition data
            alpha (float, optional): step size. Defaults to 0.1.
            gamma (float, optional): discount factor. Defaults to 0.1.

        Returns:
            int: next action - on policy method
        """
        
        # Compute td error
        next_action = self.get_action(transition.next_state)
        
        td_error = (
            transition.reward + 
            self.gamma * self.Q[transition.next_state][next_action] - 
            self.Q[transition.current_state][transition.current_action]
        )

        # Update action value
        self.Q[transition.current_state][transition.current_action] += self.alpha * td_error
        
        return next_action
        
    def get_action(self, state: Tuple) -> int:
        """Get action from epsilon greedy policy.

        Args:
            state (Tuple): q-table state index

        Returns:
            int: action
        """
        return rl_utility.epsilon_greedy(self.Q, state, self.action_count, self.epsilon)
