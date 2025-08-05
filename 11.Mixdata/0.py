import pandas as pd

a = pd.read_csv("Scikit-Learn\\ab.csv")
print(a)
print(a.isnull().sum())
print(a.info())

a["A"].fillna(a["A"].mode()[0], inplace=True)
# print(b.isnull().sum())

print(a["A"].value_counts())

a["A"].replace("2+", "3", inplace=True)

a["A"] = a["A"].astype("float")
print(a.info())
