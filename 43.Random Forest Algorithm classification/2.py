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

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# bg = BaggingClassifier(n_estimators=100)
bg = BaggingClassifier(n_estimators=30, estimator=SVC)
bg.fit(x_train, y_train)
print(bg.score(x_test, y_test) * 100, bg.score(x_train, y_train) * 100)


sc = SVC()
# bg = BaggingClassifier(n_estimators=30, estimator=SVC)
sc.fit(x_train, y_train)
print(sc.score(x_test, y_test) * 100, sc.score(x_train, y_train) * 100)

rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(x_train, y_train)
print(bg.score(x_test, y_test) * 100, rfc.score(x_train, y_train) * 100)
