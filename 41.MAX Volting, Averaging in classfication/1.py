import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

x, y = make_moons(n_samples=1000, noise=0.2)
print()

a = {"x1": x[:, 0], "x2": x[:, 1], "output": y}
new = pd.DataFrame(a)
print(new)

sns.scatterplot(x="x1", y="x2", data=new, hue=y)
plt.show()

x_a = new.iloc[:, :-1]
y_a = new["output"]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x_a, y_a, test_size=0.2, random_state=42
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
print(dt.score(x_test, y_test) * 100, dt.score(x_train, y_train) * 100)

sc = SVC()
sc.fit(x_train, y_train)
print(sc.score(x_test, y_test) * 100, sc.score(x_train, y_train) * 100)

gnb = GaussianNB()
gnb.fit(x_train, y_train)
print(gnb.score(x_test, y_test) * 100, gnb.score(x_train, y_train) * 100)


print("============")

from sklearn.ensemble import VotingClassifier

li = [("D_tree", DecisionTreeClassifier()), ("sc", SVC()), ("gb", GaussianNB())]
vc = VotingClassifier(li)
vc.fit(x_train, y_train)
print(vc.score(x_test, y_test) * 100, vc.score(x_train, y_train) * 100)


prd = pd.DataFrame(
    {
        "dt": dt.predict(x_test),
        "svm": sc.predict(x_test),
        "gnb": gnb.predict(x_test),
        "v": vc.predict(x_test),
    }
)
print(prd)
