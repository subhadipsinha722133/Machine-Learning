import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

a = pd.read_csv("Scikit-Learn/cancer_csv_file/survey lung cancer.csv")
print(a)

a = a.drop(columns="LUNG_CANCER")
print(a.isnull().sum())

n = a.select_dtypes(include="object")
m = pd.get_dummies(n, drop_first="True")

b = n.keys()
bb = m.keys()

from sklearn.preprocessing import OneHotEncoder

oe = OneHotEncoder(drop="first")
k = pd.DataFrame(oe.fit_transform(n).toarray())
print(k)
