import sklearn
from sklearn.datasets import load_iris
import pandas as pd

a = pd.read_csv("Scikit-Learn\\ff.csv")
# print((a))
print(a.head())

X = a.drop("DMC", axis=1)
y = a["DMC"]

print(X.shape)

print(y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(X_train)
