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

sns.set()
miss_value_per = pd.DataFrame((df.isnull().sum() / len(df)) * 100)
miss_value_per.plot(
    kind="bar", title="Missing value in percentage", ylabel="percentage"
)
plt.show()


print(f"size of dataset:{df.shape}")
df.drop(["body"], axis=1, inplace=True)
print(f"Size of dataset after droping :{df.shape}")
