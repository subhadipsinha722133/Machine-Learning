import numpy as np
import pandas as pd

aa = pd.read_csv("Scikit-Learn\\19.Logistic Regression\\placement.csv")
print(aa.head())
print()
print(aa.shape)
print()

print(aa.info())
print()

print(aa.isnull().sum())
print("........................")

df = aa.iloc[:, 1:]
print(df.head())

import matplotlib.pyplot as plt

plt.scatter(df["cgpa"], df["iq"], c=df["placement"])
plt.show()
print("...................")

X = df.iloc[:, 0:2]
y = df.iloc[:, -1]
print(X)
print(X.shape)
print(".../////////////////////////////.....")
print(y)
print(y.shape)


print("\\\\")

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

print(X_train)
print()
print(y_train)

print(X_test)
print("||||||||||||||||||||||")
print(y_test)


print("[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]")

from sklearn.preprocessing import StandardScaler

scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
print(X_train)

X_test = scalar.transform(X_test)
print(X_test)

print()

from sklearn.linear_model import LogisticRegression

lor = LogisticRegression()
print(lor.fit(X_train, y_train))

print()

y_pred = lor.predict(X_test)
print(y_pred)
print(y_test)

print()
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))
