import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('50_Startups.csv')



X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]


