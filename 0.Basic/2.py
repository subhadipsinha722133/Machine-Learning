import pandas as pd

a = pd.read_csv("Scikit-Learn//ab.csv")
print(a)

print(a.shape)

print(a.isnull().sum())
print(a.isnull().sum().sum())

print(a.isnull().sum() / a.shape[0] * 100)

print(a.isnull().sum().sum() / (a.shape[0] * a.shape[1]) * 100)

import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(a.isnull())
plt.show()

mm = a.dropna()

print(mm.isnull().sum())
