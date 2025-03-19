from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from Forgetters.BaseForgetter import BaseForgetter
class ABClassifierForgetter(AdaBoostClassifier, BaseForgetter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  
        self.forget_strategies = ["test"]
        self.predictions_over_time = []
    
    def fit(self, X, y):
        self.y = y.to_numpy().ravel()
        a = super().fit(X, self.y)
        self.predictions_over_time = list(self.staged_predict(X))
        return a
        
        
        
class ABRegressorForgetter(AdaBoostRegressor, BaseForgetter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  
        self.forget_strategies = ["test"]
        self.predictions_over_time = []
        
    def fit(self, X, y):
        self.y = y.to_numpy()
        a = super().fit(X, y)
        self.predictions_over_time = list(self.staged_predict(X))
        return a
    
    