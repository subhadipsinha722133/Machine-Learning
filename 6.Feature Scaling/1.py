import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df = sns.load_dataset("titanic")
print(df.head())

df2 = df[["survived", "pclass", "age", "parch"]]
print(df2.head())

df3 = df2.fillna(df2.mean())
X = df3.drop("survived", axis=1)
y = df3["survived"]
print("Shape of X = ", X.shape)
print("Shape of y = ", y.shape)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=51
)
print("Shape of X_train = ", X_train.shape)
print("Shape of y_train = ", y_train.shape)
print("Shape of X_test = ", X_test.shape)
print("Shape of y_test = ", y_test.shape)

sc = StandardScaler()
print(sc.fit(X_train))

print(X_train.describe())

print(sc.scale_)

print(sc.mean_)
print()
print()

X_train_sc = sc.transform(X_train)
X_test_sc = sc.transform(X_test)
print(X_train_sc)

print()
print()
X_train_sc = pd.DataFrame(X_train_sc, columns=["pclass", "age", "parch"])
X_test_sc = pd.DataFrame(X_test_sc, columns=["pclass", "age", "parch"])

print(X_train_sc.head())
print()
print()

# print(X_train_sc.describe())
print(X_train_sc.describe().round(2))

print()
print("==============================================")
mmc = MinMaxScaler()
mmc.fit(X_train)
X_train_mmc = mmc.transform(X_train)
X_test_mmc = mmc.transform(X_test)
print(X_train_mmc)

print()
print("==============================================")

X_train_mmc = pd.DataFrame(X_train_mmc, columns=["pclass", "age", "parch"])
X_test_mmc = pd.DataFrame(X_test_mmc, columns=["pclass", "age", "parch"])
print(X_train_mmc.describe().round(2))

import matplotlib.pyplot as plt
import seaborn as sns

sns.pairplot(X_train)
plt.show()

sns.pairplot(X_test_sc)
plt.show()

sns.pairplot(X_train_mmc)
plt.show()
