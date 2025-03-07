import numpy as np
from itertools import tee

from CompressionStrategies.BaseCompressionStrategy import BaseCompressionStrategy

class DropNForgetsClassification(BaseCompressionStrategy):
    def __init__(self, n, remember=True, limit_ratio=None, random_keeps=None, random_deletes=None):
        super().__init__(limit_ratio=limit_ratio, random_keeps=random_keeps, random_deletes=random_deletes)
        self.forget_counts = []
        self.n = n
        self.remember = remember
        
    # Same as dropUnforgettable for n=1
    def get_compression_mask(self, y_over_time, y):
        y_initial = y_over_time[0]
        forget_counts = np.zeros(len(y), dtype=int) # by default none are forgotten, all to delete
        
        last_predictions = (y == y_initial)
        for y_predicted in y_over_time[1:]:
            
            new_predictions = (y == y_predicted.ravel())
            
            # If was classified correctly last iteration, and is misclassified in the new one, it is forgotten
            new_forgotten_points = np.bitwise_and(last_predictions == 1, new_predictions == 0).astype(int)
            forget_counts = forget_counts + new_forgotten_points
            
            last_predictions = new_predictions
        
        if self.remember:
            self.forget_counts = forget_counts
            
        points_to_keep = (forget_counts > self.n)
        return points_to_keep
    
    def get_counts(self):
        return self.forget_counts