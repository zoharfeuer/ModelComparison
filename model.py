import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns

from time import time

def create_models_from_files(folder_path, model_type):
    models = []

    # Iterate over the files in the folder
    for filename in os.listdir(folder_path):
        # Check if the file is a CSV file
        if filename.endswith('.csv'):
            # Create the full file path by joining the folder path and the filename
            file_path = os.path.join(folder_path, filename)
            
            # Load the data from the CSV file
            data = pd.read_csv(file_path)

            # Strip spaces from column names
            data.columns = data.columns.str.strip()

            # Create a list of feature names by removing the target from the list of column names
            features = [col for col in data.columns]

            # Create a model name by removing the '.csv' and '_dataset' parts from the filename
            model_name = filename.replace('.csv', '').replace('_dataset', '')
            
            # Create a model instance
            model = Model(file_path, model_type, model_name, features)
            model.data = data

            # Add the model to the list
            models.append(model)

    return models

class Model:
    def __init__(self, path, model_type, model_name, features):
        self.path = path
        self.data = None
        self.model_type = model_type
        self.model_name = model_name
        self.features = features
        self.target = None
        self.index_col = None
        self.model = None
        self.performance_metrics = {}
        self.predictions = None
        self.train_test_split = None
        self.hyperparameters = {}
        self.fit_time = None

    def set_hyperparameter(self, key, value):
        self.hyperparameters[key] = value

    def encode_categorical(self):
        le = LabelEncoder()
        for col in self.data.columns:
            if self.data[col].dtype == 'object':
                if len(list(self.data[col].unique())) <= 2:
                    self.data[col] = le.fit_transform(self.data[col])
                else:
                    self.data = pd.get_dummies(self.data, columns=[col])
    
    def feature_selection(self):

        """
        RandomForest, being a tree-based model, is very good at handling a variety of data types (numerical, categorical)
        and does not make strong assumptions about the underlying data distribution. This makes RandomForest a popular choice
        for feature selection in many practical applications.
        """

        # Check if 'TopFeatures' is in hyperparameters
        if 'TopFeatures' not in self.hyperparameters:
            raise ValueError("'TopFeatures' not found in hyperparameters")
        
        self.encode_categorical()
        
        # Remove the index column from the dataframe if it exists
        if self.index_col in self.data.columns:
            self.data = self.data.drop(self.index_col, axis=1)

        X = self.data.drop(self.target, axis=1)
        y = self.data[self.target]
        
        # Define the model
        model = RandomForestClassifier()
        
        # Train the model
        model.fit(X, y)
        
        # Get feature importances
        importances = model.feature_importances_
        
        # Get the indices of the top K importances
        top_k_indices = np.argsort(importances)[::-1][:self.hyperparameters['TopFeatures']]
        
        # Get the names of the top K features
        self.features = X.columns[top_k_indices].tolist()
        
        # Update the data
        self.data = self.data[self.features + [self.target]]

    def predict(self):
        # Retrieve hyperparameters with default values
        test_ratio = self.hyperparameters.get('TestRatio')
        random_state = self.hyperparameters.get('RandomState')

        # Perform the train-test split
        X = self.data[self.features]
        y = self.data[self.target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=random_state)
        self.train_test_split = (X_train, X_test, y_train, y_test)

        # Instantiate the model
        if self.model_type == KNeighborsClassifier:
            n_neighbors = self.hyperparameters.get('Neighbors')
            self.model = self.model_type(n_neighbors=n_neighbors)
        else:
            self.model = self.model_type()

        # Fit the model and record the fit time
        start_time = time()
        self.model.fit(X_train, y_train)
        self.fit_time = time() - start_time

        # Make predictions
        self.predictions = self.model.predict(X_test)

        # Calculate performance metrics
        self.performance_metrics['accuracy'] = accuracy_score(y_test, self.predictions)
        self.performance_metrics['precision'] = precision_score(y_test, self.predictions)
        self.performance_metrics['recall'] = recall_score(y_test, self.predictions)
        self.performance_metrics['f1'] = f1_score(y_test, self.predictions)
                
    def print_model_details(self):
        print(f"Model name: {self.model_name}")
        print(f"Model type: {self.model_type}")
        print(f"Hyperparameters: {self.hyperparameters}")
        print(f"Features: {self.features}")
        print(f"Performance Metrics: {self.performance_metrics}")
        print()
    
    def run_model_pipeline(self):
        print(f"Training {self.model_name}...")
        self.feature_selection()
        self.predict()
        self.print_model_details()
    
    def visualize_metrics(self, other_model=None):
        if other_model is None:
            print("Please provide another model to compare.")
            return

        metrics = list(self.performance_metrics.keys())
        values_self = list(self.performance_metrics.values())
        values_other = list(other_model.performance_metrics.values())

        plt.bar(np.arange(len(metrics))-0.2, values_self, width=0.4, label=self.model_type)
        plt.bar(np.arange(len(metrics))+0.2, values_other, width=0.4, label=other_model.model_type)

        plt.xticks(np.arange(len(metrics)), metrics)
        plt.title(f"Performance Metrics Comparison between {self.model_name} and {other_model.model_name}")
        plt.xlabel("Metrics")
        plt.ylabel("Value")
        plt.legend()
        plt.show()


