import pandas as pd
import numpy as np

a = pd.read_csv("Scikit-Learn\\train2.csv")
print(a)
print(a.shape)
print(a.isnull().sum())

b = a.isnull().sum() / a.shape[0] * 100
print(b)

k = a.select_dtypes(include="object")
print(k.keys())

for i in k:
    a[i].fillna(a[i].mode()[0], inplace=True)
    print(i, a[i].mode()[0])

print(a.isnull().sum())
