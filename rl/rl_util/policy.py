from typing import Tuple
import numpy as np
import random

def epsilon_greedy(q_values, state: Tuple, action_count: int, epsilon = 0.1) -> int:
    p = random.random()
    if p > epsilon:
        return np.argmax(q_values[state])
    else:
        return random.randint(0, action_count - 1)
    