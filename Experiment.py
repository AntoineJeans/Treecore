import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import openml
from datetime import datetime
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

from Forgetters import BaseForgetter
from CompressionStrategies.BaseCompressionStrategy import BaseCompressionStrategy, ImplementsPointsByPriority
from sklearn.base import BaseEstimator



import warnings




class Experiment:
    def __init__(self, dataset_id : int, n_epochs : int, forgetter : BaseForgetter, 
                 compression_strategy : BaseCompressionStrategy, predictive_model : BaseEstimator, 
                 metric = balanced_accuracy_score, seed=None):
        
        if seed is not None:
            np.random.seed(seed)
        
        self.dataset_id = dataset_id
        self.n_epochs = n_epochs
        self.forgetter = forgetter
        self.compression_strategy = compression_strategy
        self.predictive_model = predictive_model
        self.metric = metric
        
        self.logging_info = {
            "date" : datetime.now().strftime("%Y-%m-%d"),
            "time" : datetime.now().strftime("%H_%M_%S"),
            "compression_strategy" : type(compression_strategy).__name__,
            "forgetter" : type(forgetter).__name__,
            "n_epochs" : n_epochs,
            "predictive_model" : type(predictive_model).__name__,
            "metric" : metric,
            
            "dataset_id" : dataset_id,
            "dataset_name" : "",
            "dataset_cardinality" : "",  
            "dataset_training_cardinality" : "",
            "dataset_number_features" : "",         
            "performance_on_whole_dataset" : "", 
                       
        }
        
        self.archive_path = f'archive/{self.logging_info["date"]}-{self.logging_info["time"]}-{self.logging_info["compression_strategy"]}/'
    
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
        
        
    def prep_data(self, test_ratio=0.2): 
        
        # loads and splits the dataset into train and test
        dataset = openml.datasets.get_dataset(self.dataset_id)
        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
        
        enc = LabelEncoder()
        y_n = enc.fit_transform(y)
        y = pd.Series(y_n)
        X = X.select_dtypes(include=['number'])
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_ratio, shuffle=True, stratify=y)
        
        self.logging_info["dataset_name"] = dataset.name
        self.logging_info["dataset_cardinality"] = str(len(y))
        self.logging_info["dataset_training_cardinality"] = str(len(self.y_train))
        self.logging_info["dataset_number_features"] = str(len(X.columns))
        
        self.logging_info["performance_on_whole_dataset"] = self.performance_on_predictive_model(self.X_train, self.y_train)     
        
    def fit_forgetter(self):
        # Fits the forgetter to the training data 
        self.forgetter.fit(self.X_train, self.y_train)
        
    def log_performance(self):
        os.makedirs(self.archive_path, exist_ok=True)
        if isinstance(self.compression_strategy, ImplementsPointsByPriority):
            self.log_performance_by_deleted_ratio()
        else: 
            self.log_performance_for_single_case()
        self.archive_info_dict()
        
            
    def performance_on_predictive_model(self, X_train, y_train, X_test=None, y_test=None):
        if X_test is None: X_test = self.X_test
        if y_test is None: y_test = self.y_test
        
        if y_train.shape[0] == 0:
            return 0
        
        if np.unique(y_train).shape[0] == 1:
            y_pred = np.full(y_test.shape, np.unique(y_train)[0])
        else:
            self.predictive_model.fit(X_train, y_train)
            y_pred = self.predictive_model.predict(X_test)
        return self.metric(y_pred, y_test)
    
    def log_performance_by_deleted_ratio(self):
        x_baseline, y_baseline = self.get_baseline_curve()
        x_c, y_c = self.get_compressed_curve()
        
        self.logging_info["Area_Between_curves"] = str(self.calculate_ABC(x_baseline, y_baseline, x_c, y_c))
        
        self.create_and_archive_curve(x_baseline, y_baseline, x_c, y_c)
    
    
    
    def log_performance_for_single_case(self):
        
        X_train_c, y_train_c = self.forgetter.transform(self.X_train, self.compression_strategy, y=self.y_train)
        score = self.performance_on_predictive_model(X_train_c, y_train_c)
        
        self.logging_info["number_of_kept_points"] = str(len(y_train_c))
        self.logging_info["performance_on_compressed"] = str(score)
        self.logging_info["baseline_performance_for_n"] = str(self.get_baseline_for_n_points(len(y_train_c)))
        
    def get_baseline_for_n_points(self, n):
        indices = np.random.choice(len(self.X_train), n, replace=False)
        return self.performance_on_predictive_model(self.X_train[indices], self.y_train[indices])
    
    def get_baseline_curve(self, N_repeats=20):
        m = len(self.X_train)
        step = int(0.05 * m)
        all_baselines = []
        for i in range(N_repeats):
            X_train_shuf, y_train_shuf = shuffle(self.X_train, self.y_train, random_state=i)
            
            scores = []
            for j in range(1, m, step):
                score = self.performance_on_predictive_model(X_train_shuf[:j], y_train_shuf[:j])
                scores.append(score)
            all_baselines.append(scores)
            
        x = range(1, m, step)
        all_baselines = np.array(all_baselines)
        y = np.mean(all_baselines, axis=0)
        
        return x, y
    
    def get_compressed_curve(self):
        _ = self.forgetter.transform(self.X_train, self.compression_strategy)
        
        m = len(self.X_train)
        step = int(0.05 * m)

        x = range(1, m, step)
        scores = []
        for n_points in x:
            mask = self.compression_strategy.keep_highest_priority(n=n_points)
            score = self.performance_on_predictive_model(self.X_train[mask], self.y_train[mask])
            scores.append(score)
           
        return x, scores
    
    def calculate_ABC(self, x_baseline, y_baseline, x_c, y_c):
        # Area between curves
        return "not done yet"
    
    def archive_info_dict(self):
        with open(self.archive_path + "exp_info.txt", "w") as f:
            for key, value in self.logging_info.items():
                f.write(f"{key}: {value}\n")

    
    def create_and_archive_curve(self, x_baseline, y_baseline, x_c, y_c):
        plt.plot(x_c, y_c, label=self.logging_info["compression_strategy"])
        plt.plot(x_baseline, y_baseline, label="Random removals")

        plt.legend()
        plt.xlabel("Nombre de points conservés")
        plt.ylabel(f'Score de {self.logging_info["metric"]}')
        plt.title(f'Accuracy test de {self.logging_info["predictive_model"]} sur données compressées de {self.logging_info["dataset_name"]}')
        plt.grid()
        
        plt.savefig(self.archive_path + "curves.png", dpi=300, bbox_inches="tight")  # Adjust DPI for better quality


 

        
        

    
    
    