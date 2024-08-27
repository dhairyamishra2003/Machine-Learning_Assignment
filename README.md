# Machine-Learning_Assignment

DOCUMENTATION OF MACHINE-LEARNING_ASSIGNMENT:-

1.Dataset Exploration(download.ipynb):-
Explanation:- 
Loading the Dataset: The load_iris() function from sklearn.datasets returns a dictionary-like object with the data and feature names. We use this to create a pandas DataFrame, making it easier to analyze. 
First Five Rows: The .head() method shows the first five rows of the DataFrame.
Dataset Shape: The .shape attribute provides the dimensions of the dataset (number of rows and columns). 
Summary Statistics: The .describe() method calculates various summary statistics (mean, standard deviation, min, max, and quartiles) for each feature.

CODE OF DATASET EXPLORATION:

from sklearn.datasets import load_iris
import pandas as pd

# Load the dataset
iris = load_iris()

# Create a DataFrame from the dataset
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Display the first five rows of the dataset
print("First five rows:")
print(df.head())

# Display the shape of the dataset
print("\nDataset shape:")
print(df.shape)

# Compute summary statistics
print("\nSummary statistics:")
print(df.describe())

2.Data Splitting(download(1).ipynb:- Explanation:- train_test_split Function: This function randomly splits the dataset into training and testing sets. test_size=0.2 specifies that 20% of the data should be allocated to the test set, and the remaining 80% will be used for training. random_state=42 ensures that the split is reproducible (you’ll get the same split each time you run the code). Printing the Number of Samples: The .shape[0] attribute of the resulting arrays gives the number of samples (rows) in each set.

CODE OF DATA SPLITTING:
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Feature matrix
y = iris.target  # Target vector

# Split the dataset into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the number of samples in the training and testing sets
print(f"Number of samples in the training set: {X_train.shape[0]}")
print(f"Number of samples in the testing set: {X_test.shape[0]}")

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Feature matrix
y = iris.target  # Target vector

# Split the dataset into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the number of samples in the training and testing sets
print(f"Number of samples in the training set: {X_train.shape[0]}")
print(f"Number of samples in the testing set: {X_test.shape[0]}")

3.Linear Regression(download(3).ipynb:-
Explanation:-
Import Libraries: Import the necessary libraries for data manipulation, model fitting, and evaluation. 
Create/Load Dataset: Create a sample dataset or load your own. Split Dataset: Divide the data into training and testing sets. 
Fit Model: Train the linear regression model on the training data. Make Predictions: Predict salaries for the test set using the trained model. 
Evaluate Performance: Calculate and print the Mean Squared Error to evaluate model performance.

CODE OF LINEAR REGRESSION:

import numpy as np 
import pandas as pd 
from sklearn.model_selection 
import train_test_split
from sklearn.linear_model 
import LinearRegression 
from sklearn.metrics import mean_squared_error

Sample dataset
data = { 
        'YearsExperience': [1.1, 1.3, 1.5, 2.0, 2.2, 3.0, 3.2, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5], 
        'Salary': [40000, 42000, 43000, 46000, 47000, 50000, 52000, 54000, 58000, 60000, 62000, 65000, 67000, 70000, 72000, 75000, 77000, 80000, 82000, 85000], 
}

Convert to DataFrame
df = pd.DataFrame(data)

Define features and target variable
X = df[['YearsExperience']] y = df['Salary']

Split the dataset into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

Initialize the Linear Regression model
model = LinearRegression()

Fit the model to the training data
model.fit(X_train, y_train)

Make predictions on the test set
y_pred = model.predict(X_test)

Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)

Print the Mean Squared Error
print(f"Mean Squared Error (MSE): {mse:.2f}")
