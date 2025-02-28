import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

a = pd.read_csv("Scikit-Learn/19.Logistic Regression/placement.csv")
print(a.head())
print(a.isnull().sum())
print(a.info())

sns.scatterplot(x="cgpa", y="iq", data=a, hue="placement")
plt.show()

# a = a.drop(columns="date")

# from sklearn.preprocessing import LabelEncoder

# le = LabelEncoder()
# a["weather"] = le.fit_transform(a["weather"])
# print(a)

x = a.iloc[:, :-1]
y = a["placement"]

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(x)
x = pd.DataFrame(sc.transform(x), columns=x.columns)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.1, random_state=56
)

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
print(dt.score(X_test, y_test) * 100)
# print(dt.score(X_train, y_train) * 100)


from sklearn.tree import plot_tree

# plt.figure(figsize=(10, 10))
plot_tree(dt)
# plt.savefig("Scikit-Learn/26.Decision Tree Algorithm/Decision_Tree")
plt.show()


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

df = {
    "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
    "splitter": ["best", "random"],
    "max_depth": [i for i in range(2, 20)],
}

gd = GridSearchCV(DecisionTreeClassifier(), param_grid=df)
gd.fit(X_train, y_train)

print(gd.best_params_)
print(gd.best_score_)

print("===================..........................................==================")

rd = RandomizedSearchCV(DecisionTreeClassifier(), param_distributions=df, n_iter=20)
rd.fit(X_train, y_train)
print(rd.best_params_)
print(rd.best_score_)
