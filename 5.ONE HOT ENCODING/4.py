import pandas as pd

data = pd.read_csv("Scikit-Learn\\5.ONE HOT ENCODING\\tips.csv")
print(data)

print(data.isnull().sum())

en = data[["sex", "smoker", "day"]]
print(en)

d = pd.get_dummies(en)
print(d)

dd = pd.get_dummies(en, drop_first=True)
print(dd)

from sklearn.preprocessing import OneHotEncoder

one_hot = OneHotEncoder(drop="first")
arr = one_hot.fit_transform(en).toarray()
print(arr)
var = pd.DataFrame(
    arr, columns=["sex_Male", "smoker_Yes", "day_Sat", "day_Sun", "day_Thur"]
)
print(var)

print(data.keys())


import pandas as pd

d = pd.read_csv("Scikit-Learn\\5.ONE HOT ENCODING\\tips.csv")
print(d)
a = d.select_dtypes(include="object")
du = pd.get_dummies(a, drop_first=True)
print(du)

kk = a.keys()
k = du.keys()

from sklearn.preprocessing import OneHotEncoder

one = OneHotEncoder(drop="first")
arr = one.fit_transform(d[kk]).toarray()
data = pd.DataFrame(arr, columns=[k])
print(data)
