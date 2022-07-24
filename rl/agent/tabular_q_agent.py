from typing import List, Tuple
from rl.agent import Agent
import numpy as np

class TabularQAgent(Agent):
    def __init__(self, obs_shape: tuple, 
                 action_count: int, 
                 terminal_states: List[Tuple] = None) -> None:
        self.__obs_shape = obs_shape
        self.__action_count = action_count
        self.__shape = obs_shape + (action_count,)
        self._terminal_states = terminal_states
        self.reset()
        
    @property
    def obs_shape(self):
        return self.__obs_shape
    
    @property
    def action_count(self):
        return self.__action_count
    
    @property
    def shape(self):
        return self.__shape
    
    @property
    def q_vals(self):
        return self._Q.copy() 
    
    def reset(self) -> None:
        # Initialize action value q(s,a) for all state-action pairs (arbitrarily)
        self._Q = self.initializer()
        # Q(terminal, :) = 0
        if self._terminal_states is not None:
            self.set_terminal_states(self._terminal_states)
        
    def initializer(self) -> np.ndarray:
        return np.zeros(shape=self.__shape)
        
    def set_terminal_states(self, terminal_states: List[Tuple]) -> None:
        assert terminal_states is not None
        
        self.__terminal_states = terminal_states
        for s in terminal_states:
            self._Q[s] = 0
            