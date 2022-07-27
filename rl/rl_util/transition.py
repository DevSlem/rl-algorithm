from dataclasses import dataclass
from typing import Any
import numpy as np
import torch

@dataclass(frozen=True)
class Transition:
    current_state: Any
    current_action: Any
    next_state: Any
    reward: float
    
    def to_tabular(self):
        transition = Transition(
            tuple(self.current_state),
            int(self.current_action),
            tuple(self.next_state),
            self.reward
        )
        return transition
    
    def to_numpy(self):
        transition = Transition(
            np.array(self.current_state),
            np.array(self.current_action),
            np.array(self.next_state),
            np.array(self.reward)
        )
        return transition
    
    def to_tensor(self, device = None, requires_grad = False):
        transition = Transition(
            torch.tensor(self.current_state, device=device, requires_grad=requires_grad),
            torch.tensor(self.current_action, device=device, requires_grad=requires_grad),
            torch.tensor(self.next_state, device=device, requires_grad=requires_grad),
            torch.tensor(self.reward, device=device, requires_grad=requires_grad)
        )
        return transition