import pandas as pd

a = pd.read_csv("Scikit-Learn//price_csv_file//diamonds.csv")
print(a)

print(a.isnull().sum())
print(a.info())

aa = a.select_dtypes(include="object")

from sklearn.preprocessing import OrdinalEncoder

oe = OrdinalEncoder()
a["cut"] = oe.fit_transform(aa[["cut"]])
a["color"] = oe.fit_transform(aa[["color"]])
a["clarity"] = oe.fit_transform(aa[["clarity"]])
print(a)

x = a.drop(columns="price")
y = a["price"]

new = a.head(10)
x_new = new.drop(columns="price")
y_new = new["price"]

from sklearn.model_selection import LeaveOneOut, LeavePOut, KFold, StratifiedKFold

# lo = LeaveOneOut()
# for train, test in lo.split(x_new, y_new):
#     print(train, test)

# lp = LeavePOut(p=2)
# for train, test in lp.split(x_new, y_new):
#     print(train, test)

# kf = KFold(n_splits=5)
# for train, test in kf.split(x_new, y_new):
#     print(train, test)

# sf = StratifiedKFold(n_splits=5)  # use in classfication only
# for train, test in sf.split(x_new, y_new):
#     print(train, test)

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

p = cross_val_score(LinearRegression(), x, y, cv=5) * 100
print(p)

pp = cross_val_score(LinearRegression(), x, y, cv=KFold(n_splits=10)) * 100
print(pp)
