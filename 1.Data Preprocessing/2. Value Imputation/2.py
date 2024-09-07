import sklearn
from sklearn.datasets import load_iris


# X, y = load_iris(return_X_y=True)


# from sklearn.linear_model import LinearRegression

# model = LinearRegression()

# model.fit(X, y)
# print(model.predict(X))

# from sklearn.neighbors import KNeighborsRegressor

# mod = KNeighborsRegressor()
# mod.fit(X, y)
# print(mod.predict(X))


# import matplotlib.pyplot as plt

# pred = mod.predict(X)
# plt.scatter(pred, y)
# plt.show()


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
