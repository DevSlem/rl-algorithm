from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class Transition:
    current_state: Any
    current_action: Any
    next_state: Any
    reward: float
