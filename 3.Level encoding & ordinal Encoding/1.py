import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("Scikit-Learn\\3.Level encoding & ordinal Encoding\\tt.csv")
print(df.head())

df2 = df[["KitchenQual", "BldgType"]]
le = LabelEncoder()
print(le.fit_transform(df2["BldgType"]))

df2["BldgType_L_enc"] = le.fit_transform(df2["BldgType"])
print(df2)

print(df["BldgType"].value_counts())
print(df["KitchenQual"].value_counts())


order_Label = {"Ex": 4, "Gd": 3, "TA": 2, "Fa": 1}
df2["KitchenQual_org_enc"] = df2["KitchenQual"].map(order_Label)
print(df2)
