import pandas as pd

df = pd.read_csv("Scikit-Learn\\5.ONE HOT ENCODING\\aa.csv")
print(df)
print(df.isnull().sum())
print(df.info())

df["Car_Model"] = df["Car_Model"].fillna(df["Car_Model"].mode()[0])
print(df)


from sklearn.impute import SimpleImputer

int_o = df.select_dtypes(include="float64").columns
si = SimpleImputer(strategy="mean")
fi = si.fit_transform(df[int_o])
new = pd.DataFrame(fi, columns=[int_o])
print(new.isnull().sum())


new["Car_Model"] = df["Car_Model"]
print(new)
