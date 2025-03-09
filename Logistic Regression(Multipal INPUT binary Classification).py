import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

a = pd.read_csv("Scikit-Learn/19.Logistic Regression/placement.csv")
print(a)
print(a.isnull().sum())

sns.scatterplot(x="cgpa", y="iq", data=a, hue="placement")
plt.legend(loc=1)
plt.show()

X = a.iloc[:, :-1]
y = a["placement"]


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train, y_train)
print(lr.score(X_test, y_test) * 100)

a = int(input("enter Unnamed:-"))
b = float(input("enter cgpa :-"))
c = float(input("enter iq:-"))


print(lr.predict([[a, b, c]]))

from mlxtend.plotting import plot_decision_regions

# plot_decision_regions(x.to_numpy(), y.to_numpy(), clf=lr)
# plt.show()
from mlxtend.plotting import plot_decision_regions

plot_decision_regions(X_train, y_train.values, clf=lr, legend=2)
plt.show()
