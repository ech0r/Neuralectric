import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the dataset
dataset = pd.read_csv('csv file here')
X = dataset.iloc[:, :-1].values
Y = dataset.ilot[:, 4].values

# Encoding


# splitting the dataset into the Training and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Feature Scaling



