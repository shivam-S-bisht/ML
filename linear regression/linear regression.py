import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importng the dataset
dataset = pd.read_csv('Salary_Data.csv')


#independent and dependent variables
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values


