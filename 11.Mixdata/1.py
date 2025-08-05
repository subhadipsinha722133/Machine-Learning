import pandas as pd

a = pd.read_csv("Scikit-Learn\cust_segmentation_Data.csv")
print(a)
print(a.info())
print(a.isnull().sum())

a["Defaulted_fill"] = a["Defaulted"].fillna(a["Defaulted"].mode()[0])

aa = a.drop(columns=["Defaulted"])
print(aa.isnull().sum())

print(aa["Defaulted_fill"].value_counts())
