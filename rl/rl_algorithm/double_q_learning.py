from typing import List, Tuple
import numpy as np
from rl import TabularQAgent, Transition, epsilon_greedy

class DoubleQLearning(TabularQAgent):
    def __init__(self, obs_shape: tuple,
                 action_count: int,
                 terminal_states: List[tuple] = None,
                 epsilon = 0.1,
                 alpha = 0.1,
                 gamma = 0.9) -> None:
        super().__init__(obs_shape, action_count, terminal_states)
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        
    def reset(self) -> None:
        self._Q1 = self.initializer()
        self._Q2 = self.initializer()
        if self._terminal_states is not None:
            self.set_terminal_states(self._terminal_states)
            
    def set_terminal_states(self, terminal_states: List[Tuple]) -> None:
        assert terminal_states is not None
        
        for s in terminal_states:
            self._Q1[s] = 0
            self._Q2[s] = 0
            
    @property
    def q_vals(self):
        return self._Q1.copy()
        
    def update(self, transition: Transition) -> None:
        Q1 = self._Q1
        Q2 = self._Q2
        transition = transition.to_tabular()
        prob = np.random.rand()
        
        if prob < 0.5: # update Q1
            # get greedy action for Q1 given next state
            greedy_action = np.argmax(Q1[transition.next_state])
            # compute td error
            td_error = (
                transition.reward +
                self.gamma * Q2[transition.next_state][greedy_action] - # get Q2 given greedy action for Q1
                Q1[transition.current_state][transition.current_action]
            )
            # update Q1
            Q1[transition.current_state][transition.current_action] += self.alpha * td_error
    
        else: # update Q2
            # get greedy action for Q2 given next state
            greedy_action = np.argmax(Q2[transition.next_state])
            # compute td error
            td_error = (
                transition.reward +
                self.gamma * Q1[transition.next_state][greedy_action] - # get Q1 given greedy action for Q2
                Q2[transition.current_state][transition.current_action]
            )
            # update Q2
            Q2[transition.current_state][transition.current_action] += self.alpha * td_error
        
    def get_action(self, state) -> int:
        return epsilon_greedy(self._Q1 + self._Q2, tuple(state), self.action_count, self.epsilon)
