from typing import List
from rl import Transition, epsilon_greedy
import numpy as np

class ExpectedSarsa:
    def __init__(self, obs_shape: tuple,
                 action_count: int,
                 terminal_states: List[tuple],
                 epsilon = 0.1,
                 alpha = 0.1,
                 gamma = 0.9
                 ) -> None:
        
        self.obs_shape = obs_shape
        self.action_count = action_count
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        
        self.Q = np.zeros(obs_shape + (action_count,))
        
        
    def update(self, transition: Transition) -> None:
        # epsilon-greedy policy probailities
        greed_action_prob = 1.0 - self.epsilon + self.epsilon / self.action_count
        non_greedy_action_prob = self.epsilon / self.action_count
        # get greedy action given next state
        greedy_action = np.argmax(transition.next_state)
        # set policy distribution for current state
        policy_distribution = np.zeros(self.action_count)
        policy_distribution[greedy_action] = greed_action_prob # greedy action
        policy_distribution[np.arange(self.action_count) != greedy_action] = non_greedy_action_prob # non greedy action

        # compute expected q for td target
        expected_q = np.average(self.Q[transition.next_state], weights=policy_distribution)
        # compute td error
        td_error = transition.reward + self.gamma * expected_q - self.Q[transition.current_state][transition.current_action]
        # update q-values
        self.Q[transition.current_state][transition.current_action] += self.alpha * td_error
    
    
    def get_action(self, state: tuple) -> int:
        return epsilon_greedy(self.Q, state, self.action_count, self.epsilon)