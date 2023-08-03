# ML Model Comparison Project

## Introduction

This project allows you to compare the performance of two machine learning (ML) models in Python. The models being compared are the K-Nearest Neighbors (KNN) classifier and the Decision Tree classifier. The goal is to evaluate and contrast their performance on a given dataset using metrics such as accuracy, precision, recall, and F1-score.

## Project Overview

The project involves the following main steps:

1. Importing the necessary libraries for data manipulation, model training, and evaluation.
2. Loading the dataset and preparing it for model training.
3. Splitting the dataset into training and testing sets.
4. Training the KNN and Decision Tree models on the training data.
5. Evaluating the performance of both models on the testing data.
6. Comparing the metrics and selecting the best-performing model.

## Libraries to Import

The project uses the following libraries:

1. `numpy`: For numerical operations and array handling.
2. `pandas`: For data manipulation and preprocessing.
3. `sklearn`: For machine learning algorithms and evaluation metrics.
   - `KNeighborsClassifier`: For the KNN model.
   - `DecisionTreeClassifier`: For the Decision Tree model.
   - `RandomForestClassifier`: For the Random Forest model (mentioned in `model.py`).
   - `LabelEncoder`: For encoding categorical features.
   - `train_test_split`: For splitting the dataset.
   - `accuracy_score`, `precision_score`, `recall_score`, `f1_score`: For evaluation metrics.
4. `matplotlib`: For data visualization and plotting.
5. `kaggle`: For importing datasets from Kaggle (mentioned in `kaggle_data_importer.py`).

## Project Structure

The project consists of the following files and folders:

- `main.py`: The main script that contains the code for model comparison.
- `model.py`: A script containing the `Model` class and utility functions for model creation, including hyperparameters for splitting the test and train set.
- `kaggle_data_importer.py`: A script to import datasets from Kaggle.
- `datasets.py`: A script containing the dataset definitions (`datasets` and `data_path`).
- `utils.py`: A utility script containing helper functions for data loading and preprocessing (mentioned in the prompt).
- `visualizations.py`: A script containing functions for visualizing model performance (optional).
- `data/`: A folder to store the dataset files (CSV, Excel, etc.).
- `models/`: A folder to store the trained model files (optional).
- `README.md`: A documentation file describing the project and its steps.

## Data Preparation

Ensure you have the dataset in a suitable format (CSV, Excel, etc.) and place it in the `data/` folder. If the dataset requires any preprocessing (e.g., handling missing values, encoding categorical features), you can implement the required steps in the `utils.py` script.

## Model Training and Comparison

In the `main.py` script, import the necessary packages and load the dataset using `pandas`. Split the dataset into features and the target variable for both KNN and Decision Tree models. Next, split the data into training and testing sets using hyperparameters defined in the `model.py` script.

Train the KNN and Decision Tree models on the training data using the `fit()` method. Evaluate the performance of each model on the testing data using evaluation metrics such as accuracy, precision, recall, F1-score, etc., provided by `sklearn.metrics`.

Print the performance metrics of both models to compare their performance.

## Conclusion

The ML Model Comparison project successfully evaluates and compares the performance of the KNN and Decision Tree classifiers on a given dataset. By following the steps outlined in this README and utilizing the provided scripts, you can gain valuable insights into which model performs better for your specific task. The project provides a structured and organized approach to model comparison and can be extended to include additional models or datasets for further analysis.