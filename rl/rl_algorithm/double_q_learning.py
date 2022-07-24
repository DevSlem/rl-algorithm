from typing import List
import numpy as np
from rl import Transition, epsilon_greedy
import random

class DoubleQLearning:
    def __init__(self, obs_shape: tuple,
                 action_count: int,
                 terminal_states: List[tuple],
                 epsilon = 0.1,
                 alpha = 0.1,
                 gamma = 0.9) -> None:
        self.obs_shape = obs_shape
        self.action_count = action_count
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        
        shape = obs_shape + (action_count,)
        self.Q1 = np.zeros(shape)
        self.Q2 = np.zeros(shape)
        
        
    def update(self, transition: Transition):
        prob = random.random()
        
        if prob < 0.5: # update Q1
            # get greedy action for Q1 given next state
            greedy_action = np.argmax(self.Q1[transition.next_state])
            # compute td error
            td_error = (
                transition.reward +
                self.gamma * self.Q2[transition.next_state][greedy_action] - # get Q2 given greedy action for Q1
                self.Q1[transition.current_state][transition.current_action]
            )
            # update Q1
            self.Q1[transition.current_state][transition.current_action] += self.alpha * td_error
    
        else: # update Q2
            # get greedy action for Q2 given next state
            greedy_action = np.argmax(self.Q2[transition.next_state])
            # compute td error
            td_error = (
                transition.reward +
                self.gamma * self.Q1[transition.next_state][greedy_action] - # get Q1 given greedy action for Q2
                self.Q2[transition.current_state][transition.current_action]
            )
            # update Q2
            self.Q2[transition.current_state][transition.current_action] += self.alpha * td_error
        
        
    def get_action(self, state: tuple) -> int:
        return epsilon_greedy(self.Q1 + self.Q2, state, self.action_count, self.epsilon)