import pandas as pd

aa = pd.read_csv("Scikit-Learn/null.csv")
print(aa)
print(aa.shape)
print(aa.isnull().sum())
print(aa.info())


print(aa["2"].value_counts())
aa["2"].replace("pclass", "3", inplace=True)
aa["2"] = aa["2"].astype("float")

print("================================")


# aa["3"].fillna(aa["3"].mode()[0], inplace=True)


print(aa["3"].value_counts())
aa["3"].replace("survived", "0", inplace=True)
aa["3"] = aa["3"].astype("float")

aa["4"].fillna(aa["4"].mode()[0], inplace=True)


print(aa["6"].value_counts())
aa["6"].replace("age", "24", inplace=True)
aa["6"] = aa["6"].astype("float")


print(aa.info())
print(aa)
