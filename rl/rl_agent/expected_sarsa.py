from typing import List
from rl import TabularQAgent, Transition, epsilon_greedy, epsilon_greedy_distribution
import numpy as np

class ExpectedSarsa(TabularQAgent):
    """ On-policy Expected Sarsa """
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
        
    def update(self, transition: Transition) -> None:
        Q = self._Q
        transition = transition.to_tabular()
        
        # get epsilon-greedy policy distribution which is the behavior policy.
        policy_distribution = epsilon_greedy_distribution(Q, transition.next_state, self.action_count, self.epsilon)
        # compute expected q
        expected_q = np.average(self.Q[transition.next_state], weights=policy_distribution)
        # compute td error
        td_error = transition.reward + self.gamma * expected_q - self.Q[transition.current_state][transition.current_action]
        # update q-values
        self.Q[transition.current_state][transition.current_action] += self.alpha * td_error
    
    def get_action(self, state) -> int:
        return epsilon_greedy(self.Q, tuple(state), self.action_count, self.epsilon)