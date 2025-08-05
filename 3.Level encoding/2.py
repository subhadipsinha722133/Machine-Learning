import pandas as pd

# df = pd.read_csv("Scikit-Learn\\3.Level encoding & ordinal Encoding\\tt.csv")

df = pd.DataFrame({"name": ["khokon", "cow", "cat", "bat", "ram", "sam"]})
print(df)
print()


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

m = le.fit_transform(df["name"])
print(m)

df["en_name"] = le.fit_transform(df["name"])
print(df)

print()
