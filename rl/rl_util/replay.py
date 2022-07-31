from __future__ import annotations
from rl.rl_util import Transition

class Replay:
    def __init__(self, max_count: int = -1) -> None:
        self._transitions: list[Transition] = []
        self.__count = 0
        self.__max_count = max_count
        
    def reset(self):
        """ Reset the replay. """
        self._transitions.clear()
        self.__count = 0
        
    def add(self, transition: Transition):
        """ Add a transition. """
        self._transitions.append(transition)
        self.__count += 1
        if self.__count == self.__max_count + 1:
            self._transitions.pop(0)
    
    def sample(self):
        """ Sample from the replay. """
        return self._transitions.copy()
    
    @property
    def count(self) -> int:
        return self.__count
    
    @property
    def max_count(self) -> int:
        return self.__max_count
    
    @property
    def is_empty(self) -> bool:
        return self.count == 0
    
    @property
    def is_full(self) -> bool:
        return self.count == self.max_count