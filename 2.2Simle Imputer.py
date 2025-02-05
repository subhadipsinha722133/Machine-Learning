import pandas as pd

a = pd.read_csv("Scikit-Learn/flipkart_com-ecommerce_sample.csv")
print(a)
print(a.isnull().sum())

p = a.isnull().sum() / a.shape[0] * 100
print(p)

print(a.info())

ob = a.select_dtypes(include="object").columns
print(ob)
for i in ob:
    a[i].fillna(a[i].mode()[0], inplace=True)
print(ob.isnull().sum())


from sklearn.impute import SimpleImputer

int_o = a.select_dtypes(include="float64").columns
si = SimpleImputer(strategy="mean")
fi = si.fit_transform(a[int_o])
new = pd.DataFrame(fi, columns=[int_o])
print(new.isnull().sum())


si2 = SimpleImputer(strategy="most_frequent")
fi2 = si2.fit_transform(a[ob])
d = pd.DataFrame(fi2, columns=[ob])
print(d.isnull().sum())
