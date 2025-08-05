import pandas as pd

data = pd.read_csv("Scikit-Learn\\mm.csv")
print(data)
p = data.select_dtypes(include="object")
pp = pd.get_dummies(p, drop_first="True")
k = p.keys()
kk = pp.keys()
from sklearn.preprocessing import OneHotEncoder

oneH = OneHotEncoder(drop="first")
ooooo = oneH.fit_transform(data[k]).toarray()

new = pd.DataFrame(ooooo, columns=[kk])
print(new)


# aa = data.select_dtypes(include="object")
# bb = pd.get_dummies(aa, drop_first=True)
# a = aa.keys()
# b = bb.keys()

# from sklearn.preprocessing import OneHotEncoder

# OH = OneHotEncoder(drop="first")
# arr = OH.fit_transform(data[a]).toarray()
# new = pd.DataFrame(arr, columns=[b])
# print(new)
