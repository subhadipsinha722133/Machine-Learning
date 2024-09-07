import sklearn
from sklearn.datasets import load_iris

import pandas as pd
from sklearn.datasets import fetch_openml

df = fetch_openml("titanic", version=1, as_frame=True)["data"]
print(df.info())

print(df.isnull())

print(df.isnull().sum())


import matplotlib.pyplot as plt
import seaborn as sns

# sns.set()
# miss_value_per = pd.DataFrame((df.isnull().sum() / len(df)) * 100)
# miss_value_per.plot(
#     kind="bar", title="Missing value in percentage", ylabel="percentage"
# )
# plt.show()


print(f"size of dataset:{df.shape}")
df.drop(["body"], axis=1, inplace=True)
print(f"Size of dataset after droping :{df.shape}")


from sklearn.impute import SimpleImputer

print(f"Number of Null value in age  befor imputing:  {df.age.isnull().sum()}")
imp = SimpleImputer(strategy="mean")
df["age"] = imp.fit_transform(df[["age"]])
print(f"Number of Null value in age column after imputing: {df.age.isnull().sum()}")


def get_parameters(df):
    parameters = {}
    for col in df.columns[df.isnull().any()]:

        if (
            df[col].dtype == "float64"
            or df[col].dtype == "int64"
            or df[col].dtype == "int32"
        ):
            strategy = "mean"
        else:
            strategy = "most_frequent"

        missing_values = df[col][df[col].isnull()].values[0]
        parameters[col] = {"missing_values": missing_values, "strategy": strategy}
    return parameters


print(get_parameters(df))


df["family"] = df["sibsp"] + df["parch"]
df.loc[df["family"] > 0, "travelled_alone"] = 0
df.loc[df["family"] == 0, "travelled_alone"] = 1
df["travelled_alone"].value_counts().plot(
    title="Passanger travelled alone?", kind="bar"
)
plt.show()


from sklearn.preprocessing import OneHotEncoder

df[["female", "male"]] = OneHotEncoder().fit_transform(df[["sex"]]).toarray()
print(df[["sex", "female", "male"]])

df["sex"] = OneHotEncoder().fit_transform(df[["sex"]]).toarray()[:, 1]
print(df.head())


from sklearn.preprocessing import StandardScaler

num_cols = df.select_dtypes(include=["int64", "float64", "int32"]).columns
print(num_cols)
ss = StandardScaler()
df[num_cols] = ss.fit_transform(df[num_cols])
print(df[num_cols].describe())


from sklearn.preprocessing import MinMaxScaler

minmax = MinMaxScaler()
df[num_cols] = minmax.fit_transform(df[num_cols])
print(df[num_cols])
