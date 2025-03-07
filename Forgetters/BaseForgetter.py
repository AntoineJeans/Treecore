from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin
from itertools import tee

class BaseForgetter(ABC, BaseEstimator, TransformerMixin):
    
    def __init__(self, **_):
        pass

    @abstractmethod
    def fit(self, X, y):
        pass
    
    def transform(self, X, compression_strategy, y=None):     
        mask = compression_strategy.get_compression_mask(self.predictions_over_time, self.y)
        if y is None:
            return X[mask]
        else: return X[mask], y[mask]
    
    def get_mask(self, compression_strategy):
        return compression_strategy.get_compression_mask(self.predictions_over_time, self.y)
    
    def fit_transform(self, X, y, compression_strategy, **kwargs):
        self.fit(X, y, **kwargs)
        return self.transform(X, compression_strategy, y)

    