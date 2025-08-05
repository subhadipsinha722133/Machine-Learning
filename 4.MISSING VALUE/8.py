# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("Scikit-Learn\\null_.csv")
print(df)
print(df.isnull().sum())

from sklearn.impute import SimpleImputer as si

a = df.select_dtypes(include=["int", "float"]).columns
print(a)

mp = si(strategy="mean")

v = mp.fit_transform(df[a])

new = pd.DataFrame(v, columns=df.select_dtypes(include=["int", "float"]).columns)
print(new)
print(new.isnull().sum())
