from abc import *
from typing import Any
import rl

class Agent(metaclass=ABCMeta):
    @abstractmethod
    def get_action(self, state) -> Any:
        """ Returns an action that follows the behavior policy. """
        pass
    
    def start_episode(self) -> Any:
        """ Call this method before the start of an episode when training. """
        pass
    
    @abstractmethod
    def update(self, transition: rl.Transition) -> Any:
        """ Update the agent when training. """
        pass
    
    def end_episode(self) -> Any:
        """ Call this method after the end of an episode when training. """
        pass
    