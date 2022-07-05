from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)
class Transition:
    current_state: Tuple
    current_action: int
    next_state: Tuple
    reward: float
