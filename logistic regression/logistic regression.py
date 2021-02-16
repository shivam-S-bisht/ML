import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Social_Network_Ads.csv')


X = dataset.iloc[:, 2:-1]
y = dataset.iloc[:, -1]
