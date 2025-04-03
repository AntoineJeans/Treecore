from CompressionStrategies.DropNForgets import DropNForgetsClassification
from Forgetters.AdaptiveBoostingForgetters import ABClassifierForgetter
from sklearn.tree import DecisionTreeClassifier

from Experiment import Experiment

# mnist = 554
# blood = 1464
dataset_id = 40981 
n_epochs = 100
forgetter = ABClassifierForgetter(n_estimators=n_epochs, estimator=DecisionTreeClassifier(max_depth=2))
compression_strategy = DropNForgetsClassification(n = 1)

from sklearn.linear_model import LogisticRegression
predictive_model = LogisticRegression(C=1)

exp = Experiment(dataset_id, n_epochs, forgetter, compression_strategy, predictive_model, seed=123)

exp.prep_data()
exp.fit_forgetter()
exp.log_performance()
