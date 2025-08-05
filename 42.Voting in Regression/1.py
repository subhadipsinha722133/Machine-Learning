import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

a = pd.read_csv("Scikit-Learn/price_csv_file/house_price.csv")
print(a)

print(a.isnull().sum())
print(a.info())

aa = a.select_dtypes("object")

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
a["street"] = le.fit_transform(aa["street"])
a["city"] = le.fit_transform(aa["city"])
a["statezip"] = le.fit_transform(aa["statezip"])
a["date"] = le.fit_transform(aa["date"])


aa = a.drop(columns="country", inplace=True)
aa = a.drop(columns="price")
print(aa)

print(aa.keys())

x = aa
y = a["price"]
print(y)


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression

dt = DecisionTreeRegressor()
dt.fit(x_train, y_train)
print(dt.score(x_test, y_test) * 100, dt.score(x_train, y_train) * 100)

sc = SVR()
sc.fit(x_train, y_train)
print(sc.score(x_test, y_test) * 100, sc.score(x_train, y_train) * 100)

lig = LinearRegression()
lig.fit(x_train, y_train)
print(lig.score(x_test, y_test) * 100, lig.score(x_train, y_train) * 100)


print("============")
from sklearn.ensemble import VotingRegressor

li = [("D_tree", DecisionTreeRegressor()), ("sc", SVR()), ("li", LinearRegression())]
vc = VotingRegressor(li)
vc.fit(x_train, y_train)
print(vc.score(x_test, y_test) * 100, vc.score(x_train, y_train) * 100)


prd = pd.DataFrame(
    {
        "dt": dt.predict(x_test),
        "svm": sc.predict(x_test),
        "lr": lig.predict(x_test),
        "v": vc.predict(x_test),
    }
)
print(prd)
