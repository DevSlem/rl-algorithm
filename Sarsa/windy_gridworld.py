from typing import Tuple
import numpy as np

class WindyGridworld:
    def __init__(self) -> None:
        self.obs_shape = (7, 10)
        self.action_count = 4
        
        # wind setting
        self.wind = np.zeros(self.obs_shape + (2,), dtype=np.int32)
        self.wind[:, 3:6, 0] = -1
        self.wind[:, 6:8, 0] = -2
        self.wind[:, 8, 0] = -1
        
        self.reset()
        self.goal_state = np.array([3, 7], dtype=np.int32) # goal state
        
    def reset(self) -> np.ndarray:
        """ Reset the environment.

        Returns:
            np.ndarray: start state
        """
        self.state = np.array([3, 0], dtype=np.int32) # start state
        return self.state
        
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """ Move to next step.

        Args:
            action (int): 0: up, 1: right, 2: down, 3: left

        Returns:
            Tuple[np.ndarray, float, bool]: next state, reward, terminated
        """
        
        self.move(action)
        terminated = (self.state == self.goal_state).sum() == 2
        reward = 0 if terminated else -1
        
        return self.state, reward, terminated
    
    def convert_to_move(self, action: int) -> np.ndarray:
        assert action >= 0 and action < self.action_count
        temp = None
        if action == 0: # up
            temp = [-1, 0]
        elif action == 1: # right
            temp = [0, 1]
        elif action == 2: # down
            temp = [1, 0]
        else: # left
            temp = [0, -1]
            
        return np.array(temp, dtype=np.int32)
    
    def move(self, action: int):
        delta_pos = self.convert_to_move(action)
        state = self.state + delta_pos + self.wind[tuple(self.state)]
        state = np.clip(state, [0, 0], np.array(self.obs_shape) - 1)
        self.state = state
        