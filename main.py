import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.tree import DecisionTreeClassifier as dt
from kaggle_data_importer import get_datasets
from datasets import datasets, data_path
from model import Model
from model import create_models_from_files

get_datasets(datasets=datasets, path=data_path)

KNN_models = create_models_from_files(folder_path="data", model_type=KNN)
desctree_models=  create_models_from_files(folder_path="data", model_type=dt)

for model in KNN_models:
    model.target = model.features[-1]
    model.index_col = model.features[0]
    model.set_hyperparameter(key="TopFeatures", value=5)
    model.set_hyperparameter(key="Neighbors", value=5)
    model.set_hyperparameter(key="TestRatio", value=0.2)
    model.set_hyperparameter(key="RandomState", value=42)

for model in desctree_models:
    model.target = model.features[-1]
    model.index_col = model.features[0]
    model.set_hyperparameter(key="TopFeatures", value=5)
    model.set_hyperparameter(key="Neighbors", value=5)
    model.set_hyperparameter(key="TestRatio", value=0.2)
    model.set_hyperparameter(key="RandomState", value=42)

for KNN_models, desctree_models in zip(KNN_models, desctree_models):
    KNN_models.run_model_pipeline()
    desctree_models.run_model_pipeline()
    KNN_models.visualize_metrics(other_model=desctree_models)
   
