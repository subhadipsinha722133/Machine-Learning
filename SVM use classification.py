import pandas as pd

a = pd.read_csv("Scikit-Learn\\flawer_csv file\\flower2.csv")
print(a)
print(a.isnull().sum())

import matplotlib.pyplot as plt
import seaborn as sns

sns.pairplot(data=a)
plt.show()

x = a.iloc[:, :-1]
y = a["species"]

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(x)
x = pd.DataFrame(ss.transform(x), columns=x.columns)

from sklearn.model_selection import train_test_split

X_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

from sklearn.svm import SVC

sc = SVC(kernel="linear")
sc.fit(X_train, y_train)
print(sc.score(x_test, y_test) * 100)
print(sc.score(X_train, y_train) * 100)  # chack overfetting


sc = SVC(kernel="poly")
sc.fit(X_train, y_train)
print(sc.score(x_test, y_test) * 100)
print(sc.score(X_train, y_train) * 100)  # chack overfetting


# sc = SVC(kernel="precomputed")
# sc.fit(X_train, y_train)
# print(sc.score(x_test, y_test) * 100)
# print(sc.score(X_train, y_train) * 100)  # chack overfetting


sc = SVC(kernel="rbf")
sc.fit(X_train, y_train)
print(sc.score(x_test, y_test) * 100)
print(sc.score(X_train, y_train) * 100)  # chack overfetting


sc = SVC(kernel="sigmoid")
sc.fit(X_train, y_train)
print(sc.score(x_test, y_test) * 100)
print(sc.score(X_train, y_train) * 100)  # chack overfetting
