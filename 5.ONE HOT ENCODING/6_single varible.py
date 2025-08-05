import pandas as pd

l = pd.read_csv(r"C:\Users\subhadip sinha\OneDrive\CODING\Jupyter notebook\drug.csv")
print(l)

print(l.info())
a = l["sex"]
print(a)

from sklearn.preprocessing import OneHotEncoder

p = pd.get_dummies(a, drop_first=True)
print(p)
pp = p.keys()
one = OneHotEncoder(drop="first")
q = one.fit_transform(l[["sex"]]).toarray()
nw = pd.DataFrame(q, columns=[pp])
print(nw)
