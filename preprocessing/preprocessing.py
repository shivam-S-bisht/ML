import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# importing the dataset
dataset = pd.read_csv('Data.csv')


# #converting categorical data variables to dummy variables
dataset = pd.get_dummies(data = dataset, columns=['Country', 'Purchased'])
print(dataset)
