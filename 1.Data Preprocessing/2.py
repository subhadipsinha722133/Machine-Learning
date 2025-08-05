import pandas as pd

a = pd.read_csv("Scikit-Learn\\ff.csv")
print(a.isnull().sum())
print(a.head())
print(a.info())

import seaborn as sns
import matplotlib.pyplot as plt

# sns.pairplot(a)
plt.show()

from sklearn.preprocessing import OrdinalEncoder

oe = OrdinalEncoder()
a["month"] = oe.fit_transform(a[["month"]])
a["day"] = oe.fit_transform(a[["day"]])

print(a.info())


x = a.drop("DMC", axis=1)
y = a["DMC"]
from sklearn.preprocessing import StandardScaler

st = StandardScaler()
st.fit(x)
x = pd.DataFrame(st.transform(x), columns=x.columns)
print(x)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=70
)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)
print(lr.score(X_test, y_test) * 100)

from sklearn.svm import SVR

lo = SVR()
lo.fit(X_train, y_train)
print(lo.score(X_test, y_test) * 100)

from sklearn.tree import DecisionTreeRegressor, plot_tree

nn = DecisionTreeRegressor()
nn.fit(X_train, y_train)
print(nn.score(X_test, y_test) * 100)

# plot_tree(nn)
# plt.show()

from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor()
knn.fit(X_train, y_train)
print(knn.score(X_test, y_test) * 100)
