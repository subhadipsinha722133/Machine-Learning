import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


a = pd.read_csv("Scikit-Learn/bb.csv")
print(a.isnull().sum())

import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(data=a)
plt.show()
sns.scatterplot(x="TV", y="Sales", data=a)
# plt.plot(a["TV"])
plt.show()

from sklearn.model_selection import train_test_split

x = a.iloc[:, :-1]
# X = a["TV"]
y = a["Sales"]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

bg = BaggingRegressor(n_estimators=100)
# bg = BaggingClassifier(n_estimators=30, estimator=SVC)
bg.fit(x_train, y_train)
print(bg.score(x_test, y_test) * 100, bg.score(x_train, y_train) * 100)


sr = SVR()
# bg = BaggingClassifier(n_estimators=30, estimator=SVC)
sr.fit(x_train, y_train)
print(sr.score(x_test, y_test) * 100, sr.score(x_train, y_train) * 100)

rfr = RandomForestRegressor(n_estimators=100)
rfr.fit(x_train, y_train)
print(rfr.score(x_test, y_test) * 100, rfr.score(x_train, y_train) * 100)
