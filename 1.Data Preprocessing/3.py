import pandas as pd

a = pd.read_csv("Scikit-Learn/cust_segmentation_Data.csv")
print(a)

print(a.isnull().sum())

print(a.info())

from sklearn.impute import SimpleImputer


k = a.select_dtypes(include="float64")
kk = k.keys()
sim = SimpleImputer(strategy="mean")
w = pd.DataFrame(sim.fit_transform(a[kk]), columns=[kk])
print(w)
