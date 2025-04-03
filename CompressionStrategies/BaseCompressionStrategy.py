import numpy as np
from abc import ABC, abstractmethod
class BaseCompressionStrategy:
    def __init__(self, limit_ratio=1, random_keeps=0, random_deletes=0):
        self.limit_ratio=limit_ratio
        self.random_keeps=random_keeps
        self.random_deletes=random_deletes
        
        
    def get_compression_mask(self, y_over_time, y):
        mask = self.get_strategy_mask(y_over_time, y)
        np.random.seed(42)
        random_keeps_mask = np.random.rand(*mask.shape) < self.random_keeps
        return mask | random_keeps_mask
    
    def get_strategy_mask(self, y_over_time, y):
        pass
    
class ImplementsPointsByPriority(ABC):
    @abstractmethod
    def get_priorities(self):
        pass
    
    def keep_highest_priority(self, n):
        point_order = self.get_priorities()
        n_points = len(point_order)
        
        # if input n is a percentage (double), we keep that percentage of the dataset
        if type(n) == float:
            n = min(int(n*n_points), n_points)
            
        point_ids_to_keep = point_order[:n]
        mask = [(i in point_ids_to_keep) for i in range(n_points)]       
        return np.array(mask)