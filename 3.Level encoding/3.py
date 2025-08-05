import pandas as pd

a = pd.read_csv("Scikit-Learn/german.csv")
print(a)

print(a.isnull().sum().sum())

print(a["Purpose"].unique())
from sklearn.preprocessing import LabelEncoder

la = LabelEncoder()
l = la.fit(a["Purpose"])
print(l)

# a["Purpose"] = la.transform(a["Purpose"])
# print(a)

# from sklearn.preprocessing import LabelEncoder

# la = LabelEncoder()

a["Purpose"] = la.fit_transform(a[["Purpose"]])
print(a)
