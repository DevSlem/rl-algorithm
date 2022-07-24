from typing import List, Tuple
from rl import TabularQAgent, Transition, epsilon_greedy

class Sarsa(TabularQAgent):
    def __init__(self, obs_shape: tuple, 
                 action_count: int, 
                 terminal_states: List[Tuple] = None, 
                 epsilon = 0.1,
                 alpha = 0.1,
                 gamma = 0.9) -> None:
        super().__init__(obs_shape, action_count, terminal_states)
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma  
        self.cur_transition = None # current transition to update q values for it
        
    def start_episode(self):
        self.cur_transition = None
    
    def update(self, transition: Transition):
        # if you don't have a next action, skip the update rule until the next update call
        if self.cur_transition is None:
            self.cur_transition = transition.to_tabular()
            pass
        
        Q = self._Q
        transition = transition.to_tabular() # next transition
        
        # get a next aciton that follows the current policy
        next_action = transition.current_action
        # compute td error
        td_error = (
            self.cur_transition.reward + 
            self.gamma * Q[self.cur_transition.next_state][next_action] - 
            Q[self.cur_transition.current_state][self.cur_transition.current_action]
        )
        # update q-value
        Q[self.cur_transition.current_state][self.cur_transition.current_action] += self.alpha * td_error
        
        # update the current transition
        self.cur_transition = transition
        
    def get_action(self, state) -> int:
        return epsilon_greedy(self._Q, tuple(state), self.action_count, self.epsilon)
    