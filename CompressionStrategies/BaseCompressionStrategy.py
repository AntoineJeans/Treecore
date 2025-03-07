from abc import ABC, abstractmethod
class BaseCompressionStrategy:
    def __init__(self, limit_ratio, random_keeps, random_deletes):
        self.limit_ratio=limit_ratio
        self.random_keeps=random_keeps
        self.random_deletes=random_deletes
        
        
    def get_compression_mask(self, X, y_over_time, y):
        super().get_compression_mask(X, y_over_time, y)