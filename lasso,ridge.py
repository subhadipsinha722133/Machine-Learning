import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

a = pd.read_csv("Scikit-Learn/churn_Data.csv")
print(a)

plt.figure(figsize=(10, 5))
sns.heatmap(data=a.corr(), annot=True)  # use only int and float
plt.show()

print(a.isnull().sum())

x = a.iloc[:, :-1]
y = a["churn"]

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(x)
x = pd.DataFrame(sc.transform(x), columns=x.columns)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

lr = LinearRegression()
lr.fit(X_train, y_train)
print(lr.score(X_test, y_test) * 100)

print(mean_squared_error(y_test, lr.predict(X_test)))
print(mean_absolute_error(y_test, lr.predict(X_test)))
print(np.sqrt(mean_squared_error(y_test, lr.predict(X_test))))

print(lr.coef_)

plt.bar(x.columns, lr.coef_)
plt.xlabel("colums")
plt.ylabel("coef")
plt.show()

print("LASSO================================================================")
la = Lasso(alpha=10)
la.fit(X_train, y_train)
print(la.score(X_test, y_test) * 100)

print(mean_squared_error(y_test, la.predict(X_test)))
print(mean_absolute_error(y_test, la.predict(X_test)))
print(np.sqrt(mean_squared_error(y_test, la.predict(X_test))))

plt.bar(x.columns, la.coef_)
plt.xlabel("colums")
plt.ylabel("coef")
plt.show()

print("RIDGE==============================================================")
ri = Ridge(alpha=10)
ri.fit(X_train, y_train)
print(la.score(X_test, y_test) * 100)

print(mean_squared_error(y_test, ri.predict(X_test)))
print(mean_absolute_error(y_test, ri.predict(X_test)))
print(np.sqrt(mean_squared_error(y_test, ri.predict(X_test))))

plt.bar(x.columns, ri.coef_)
plt.xlabel("colums")
plt.ylabel("coef")
plt.show()
