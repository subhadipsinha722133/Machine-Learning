import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_openml


titanic_data = fetch_openml("titanic", version=1, as_frame=True)

df = titanic_data["data"]
df["survived"] = titanic_data["target"]
print(df.head())

# sns.countplot(x="survived", data=df)
# sns.countplot(x="survived", hue="sex", data=df)
# sns.countplot(x="survived", hue="pclass", data=df)

# df["age"].plot.hist()
plt.show()
print(df.info())

print(df.isnull().sum())

# miss_vals = pd.DataFrame(df.isnull().sum() / len(df) * 100)
# miss_vals.plot(kind="bar", title="missing values in percentage", ylabel="percentage")
# plt.show()

df["family"] = df["sibsp"] + df["parch"]
df.loc[df["family"] > 0, "travelled_alone"] = 0
df.loc[df["family"] == 0, "travelled_alone"] = 1

print(df["family"].head())

df.drop(["sibsp", "parch"], axis=1, inplace=True)
sns.countplot(x="travelled_alone", data=df)
plt.title("Number of Passengers Travelling Alone")
plt.show()

print(df.head())

df.drop(["name", "ticket", "home.dest"], axis=1, inplace=True)
print(df.head())

df.drop(["cabin", "body", "boat"], axis=1, inplace=True)
print(df.head())


sex = pd.get_dummies(df["sex"])
print(sex)


sex = pd.get_dummies(df["sex"], drop_first=True)
print(sex)

df["sex"] = sex
print(df.isnull().sum())


from sklearn.impute import SimpleImputer

imp_mean = SimpleImputer(strategy="mean")

df["age"] = imp_mean.fit_transform(df[["age"]])
df["fare"] = imp_mean.fit_transform(df[["fare"]])
print(df.isnull().sum())

print(df.head())


embark = pd.get_dummies(df["embarked"], drop_first=True)
print(embark)


df.drop(["embarked"], axis=1, inplace=True)
df = pd.concat([df, embark], axis=1)

print(df.head())

X = df.drop(["survived"], axis=1)
print(X.head())


y = df["survived"]
print(y.head())

from sklearn.model_selection import train_test_split

x_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

print(x_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

from sklearn.linear_model import LogisticRegression

mod = LogisticRegression()
mod.fit(x_train, y_train)

pred = mod.predict(X_test)
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, pred))

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, pred))
