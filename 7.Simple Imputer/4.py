import pandas as pd

a = pd.read_csv("Scikit-Learn/flipkart_com-ecommerce_sample.csv")
print(a)
print(a.isnull().sum())
print(a.info())


b = a.select_dtypes(include="object")
bb = b.keys()
c = a.select_dtypes(include="float64")
cc = c.keys()

from sklearn.impute import SimpleImputer

si = SimpleImputer(strategy="most_frequent")
s = si.fit_transform(a[bb])
new1 = pd.DataFrame(s, columns=[bb])
print(new1)

si2 = SimpleImputer(strategy="mean")
s1 = si2.fit_transform(a[cc])
new2 = pd.DataFrame(s1, columns=[cc])
print(new2)
