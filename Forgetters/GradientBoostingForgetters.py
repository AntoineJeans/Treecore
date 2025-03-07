from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from Forgetters.BaseForgetter import BaseForgetter
class GBClassifierForgetter(GradientBoostingClassifier, BaseForgetter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  
        self.forget_strategies = ["test"]
        self.predictions_over_time = []
    
    def fit(self, X, y):
        a = super().fit(X, y)
        self.predictions_over_time = list(self._staged_raw_predict(X, check_input=True))
        return a
    
class GBRegressorForgetter(GradientBoostingRegressor, BaseForgetter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  
        self.forget_strategies = ["test"]
        self.predictions_over_time = []
        
    def fit(self, X, y):
        a = super().fit(X, y)
        self.predictions_over_time = list(self._staged_raw_predict(X, check_input=True))
        self.y = y.to_numpy()
        return a