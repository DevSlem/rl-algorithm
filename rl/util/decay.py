from abc import ABCMeta, abstractmethod

class Decay(metaclass=ABCMeta):
    def __init__(self, start_val: float, end_val: float, end_step: int) -> None:
        self.start_val = start_val
        self.end_val = end_val
        self.end_step = end_step
        self.current_step = 0
    
    def step(self) -> float:
        """ Returns a value for the current step and then move to next step. """
        val = self.get_current_value()
        self.current_step += 1
        return val
    
    @abstractmethod
    def get_current_value(self) -> float:
        pass
    
class NoDecay(Decay):
    def __init__(self, value) -> None:
        super().__init__(value, 0, 0)
        
    def get_current_value(self) -> float:
        return self.start_val
    
class LinearDecay(Decay):
    def __init__(self, start_val: float, end_val: float, end_step: int) -> None:
        super().__init__(start_val, end_val, end_step)
        
    def get_current_value(self) -> float:
        slope = (self.end_val - self.start_val) / self.end_step
        val = max(slope * self.current_step + self.start_val, self.end_val)
        return val