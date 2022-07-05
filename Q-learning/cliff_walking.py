from typing import Tuple
import numpy as np

class CliffWalking:
    
    def __init__(self) -> None:
        self.obs_shape = (4, 12)
        self.action_count = 4
        self.start_state = (self.obs_shape[0] - 1, 0)
        self.goal_state = (self.obs_shape[0] - 1, self.obs_shape[1] - 1)
        
        self.reset()
        
    def reset(self) -> Tuple:
        """ Reset the environment.

        Returns:
            Tuple: start state
        """
        
        self.state = np.array(self.start_state, dtype=np.int32)
        return self.start_state
        
    def step(self, action: int) -> Tuple[Tuple, float, bool]:
        """ Move to next step.

        Args:
            action (int): 0: up, 1: right: 2: down, 3: left

        Returns:
            Tuple[Tuple, float, bool]: next state, reward, terminated
        """
        
        self.move(action)
        terminated = False
        reward = -1.0
        if self.is_falling_off_cliff():
            self.reset()
            reward = -100.0
        elif self.is_goal_state():
            reward = 0.0
            terminated = True
            
        return tuple(self.state), reward, terminated
    
    def move(self, action: int):
        move = self.convert_to_move(action)
        self.state += move
        self.state = np.clip(self.state, [0, 0], np.array(self.obs_shape) - 1)
        
    def is_falling_off_cliff(self):
        return self.state[0] == 3 and self.state[1] >= 1 and self.state[1] <= 10
    
    @property
    def cliff_area(self):
        area = np.zeros(self.obs_shape, dtype=np.bool8)
        area[3, 1:11] = True
        return area
    
    def is_goal_state(self):
        return self.state[0] == self.goal_state[0] and self.state[1] == self.goal_state[1]
    
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