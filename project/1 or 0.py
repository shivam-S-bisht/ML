import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Data2.csv')

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]
