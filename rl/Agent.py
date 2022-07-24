from abc import *
from typing import Any, List, Tuple
import rl
import numpy as np

class Agent(metaclass=ABCMeta):
    @abstractmethod
    def get_action(self, state) -> Any:
        """ Returns an action that follows the behavior policy. """
        pass
    
    def start_episode(self):
        """ Call this method before episode start. """
        pass
    
    @abstractmethod
    def update(self, transition: rl.Transition) -> Any:
        """ Update the agent. """
        pass
    
    def end_episode(self):
        """ Call this method after episode end. """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        pass
    
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
            