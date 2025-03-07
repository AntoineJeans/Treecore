import pandas as pd
import os
import pathlib
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

class DatasetLoader():
    def __init__(self, filename, y_column, classification=True):
        path = pathlib.Path(filename)
        if not path.exists():
            raise ValueError("Le fichier n'existe pas")
        
        
        self.filename = filename
        self.y_column = y_column
        self.classification = classification
        self.df = pd.read_csv(self.filename)
    
    def get_df(self):
        return self.df
    
    def keep_only_numerical_columns(self):
        self.df = self.df.select_dtypes(include=['number'])
        
    def normalize_numerical_columns(self):
        
        scaler = StandardScaler()
        y = self.df[self.y_column]
        self.df = pd.DataFrame(scaler.fit_transform(self.df), columns=self.df.columns)
        self.df[self.y_column] = y
        
    def drop_column(self, name):
        self.df.drop(columns=[name], inplace=True)
    
    def read_split(self, test_size=0.2, shuffle=True):
        y = self.df[self.y_column]
        X = self.df.drop(columns=[self.y_column], inplace=False)
                
        if self.classification:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=shuffle, stratify=y)
        else: 
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=shuffle)
            
        return X_train, X_test, y_train, y_test