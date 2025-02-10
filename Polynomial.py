import pandas as pd
import matplotlib.pyplot as plt

a = pd.read_csv("Scikit-Learn/china_gdp.csv")
print(a)

print(a.info())


x = a[["Year"]]  # multi dimensional array
y = a["Value"]

from sklearn.preprocessing import PolynomialFeatures

pf = PolynomialFeatures(degree=3)
pf.fit(x)
x = pf.transform(x)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)

print(lr.score(X_test, y_test) * 100)


prd = lr.predict(x)
plt.scatter(a["Year"], a["Value"])
plt.plot(a["Year"], prd, c="red")
plt.xlabel("Year")
plt.ylabel("Value")
plt.show()
