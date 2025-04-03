import numpy as np
from itertools import tee

from CompressionStrategies.BaseCompressionStrategy import BaseCompressionStrategy, ImplementsPointsByPriority

class DropNForgetsClassification(BaseCompressionStrategy, ImplementsPointsByPriority):
    def __init__(self, n, allow_cache = False, **kwargs):
        super().__init__(**kwargs)
        self.n = n
        self.allow_cache = allow_cache
    
    def calculate_forget_counts(self, y_over_time, y):
        y_initial = y_over_time[0]
        forget_counts = np.zeros(len(y), dtype=int) # by default none are forgotten, all to delete
        
        last_predictions = (y == y_initial)
        for y_predicted in y_over_time[1:]:
            
            new_predictions = (y == y_predicted.ravel())
            
            # If was classified correctly last iteration, and is misclassified in the new one, it is forgotten
            new_forgotten_points = np.bitwise_and(last_predictions == 1, new_predictions == 0).astype(int)
            forget_counts = forget_counts + new_forgotten_points
            last_predictions = new_predictions

        self.forget_counts = forget_counts
        return forget_counts
        
        
    # Same as dropUnforgettable for n=1
    def get_strategy_mask(self, y_over_time, y):
        return self.calculate_forget_counts(y_over_time, y) > self.n
    
    def get_priorities(self):
        return np.argsort(self.forget_counts)

    
