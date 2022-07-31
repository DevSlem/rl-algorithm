import numpy as np
from rl.rl_util import Replay

class OnPolicyReplay(Replay):
    def __init__(self, max_count: int = -1) -> None:
        super().__init__(max_count)
    
    def sample(self):
        """ Sample from the replay and reset one. """
        temp = self._transitions.copy()
        self.reset()
        return temp