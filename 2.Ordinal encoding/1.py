import pandas as pd

df = pd.DataFrame({"size": ["s", "t", "u", "h", "l", "m", "xl", "m"]})
print(df)

# ord_data = [["s", "t", "u", "h", "l", "m", "xl", "m"]]

from sklearn.preprocessing import OrdinalEncoder

o = OrdinalEncoder()
o.fit(df[["size"]])
a = o.transform(df[["size"]])

print(a)
df["new_size"] = a

ord_data = {"s": 4, "t": 8, "u": 0, "h": 9, "l": 0, "m": 1, "xl": 5, "m": 2}
df["size_map"] = df["size"].map(ord_data)
print(df)
