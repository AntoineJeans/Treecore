import numpy as np
from itertools import tee

from CompressionStrategies.BaseCompressionStrategy import BaseCompressionStrategy


class DropUnforgettableClassification(BaseCompressionStrategy):
    def __init__(self, limit_ratio=None, random_keeps=None, random_deletes=None):
        super().__init__(limit_ratio=limit_ratio, random_keeps=random_keeps, random_deletes=random_deletes)
        
    def get_strategy_mask(self, y_over_time, y):
        y_initial = y_over_time[0]
        has_been_forgotten = np.zeros(len(y), dtype=bool) # by default none are forgotten, all to delete
        
        last_predictions = (y == y_initial)
        for y_predicted in y_over_time[1:]:
            
            new_predictions = (y == y_predicted.ravel())
            
            # If was classified correctly last iteration, and is misclassified in the new one, it is forgotten
            new_forgotten_points = np.bitwise_and(last_predictions == 1, new_predictions == 0)
            has_been_forgotten = has_been_forgotten | new_forgotten_points
            
            last_predictions = new_predictions
        
        points_to_keep = has_been_forgotten
        return points_to_keep
    


class DropUnforgettableRegression(BaseCompressionStrategy):
    def __init__(self, epsilon, limit_ratio=None, random_keeps=None, random_deletes=None):
        super().__init__(limit_ratio=limit_ratio, random_keeps=random_keeps, random_deletes=random_deletes)
        self.epsilon = epsilon # The Z score 

        
    def get_strategy_mask(self, y_over_time, y):
        y_initial = y_over_time[0]
        
        has_been_forgotten = np.zeros(len(y), dtype=bool) # by default forget none
        
        # We remember the closest predictions through the whole history 
        best_prediction_gaps_to_date = np.abs(y - y_initial.ravel())  
        sigma = np.std(y)
        
        for y_predicted in y_over_time[1:]:
               
            new_predictions_gaps = np.abs(y - y_predicted.ravel())
            
            # If the current prediction is self.Z_cutoff standard deviations worse 
            # than the best prediction in history, we consider it forgotten.
            # (new_predictions_gaps - best_prediction_gaps_to_date) could be negative, that's fine
            new_forgotten_points = ((new_predictions_gaps - best_prediction_gaps_to_date)/sigma) > self.epsilon
                                              
            # updating best prediction gaps to date
            best_prediction_gaps_to_date = np.minimum(new_predictions_gaps, best_prediction_gaps_to_date)
            
            has_been_forgotten = has_been_forgotten | new_forgotten_points

        # returns the points to keep
        return has_been_forgotten
    
    def get_epsilons(self, y_over_time, y):
        y_initial = y_over_time[0]
        
        worst_epsilons = np.zeros(len(y), dtype=np.float64) # by default forget none
        
        # We remember the closest predictions through the whole history 
        best_prediction_gaps_to_date = np.abs(y - y_initial.ravel())  
        sigma = np.std(y)
        
        for y_predicted in y_over_time[1:]:
            new_predictions_gaps = np.abs(y - y_predicted.ravel())
            
            # (new_predictions_gaps - best_prediction_gaps_to_date) could be negative, that's fine
            new_epsilons = ((new_predictions_gaps - best_prediction_gaps_to_date)/sigma) 
            worst_epsilons = np.max(worst_epsilons, new_epsilons)
                     
            # updating best prediction gaps to date
            best_prediction_gaps_to_date = np.minimum(new_predictions_gaps, best_prediction_gaps_to_date)
            
        return worst_epsilons