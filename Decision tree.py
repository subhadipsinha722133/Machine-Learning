import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("Scikit-Learn\\Decision Tree Algorithm\\Drug.csv")
print(df.head)


print(df.isna().sum())  # Check for missing values in the dataset

print(df.duplicated())

print(df[df.duplicated()])

print(df.info())

# x = df.Sex.value_counts()
# print(x)
# p = sns.countplot(data=df, x="Sex")
# plt.show()


# x = df.Drug.value_counts()
# print(x)
# p = sns.countplot(data=df, x="Drug")
# plt.show()


print(df["Drug"].unique())


plt.figure(figsize=(10, 10))
sns.distplot(
    df[df["Drug"] == "drugY"]["Age"],
    color="green",
)
sns.distplot(
    df[df["Drug"] == "drugX"]["Age"],
    color="red",
)
sns.distplot(
    df[df["Drug"] == "drugA"]["Age"],
    color="black",
)
sns.distplot(
    df[df["Drug"] == "drugB"]["Age"],
    color="orange",
)
sns.distplot(
    df[df["Drug"] == "drugC"]["Age"],
    color="blue",
)
plt.title("Age Vs Drug Class")
plt.show()


from sklearn.preprocessing import OrdinalEncoder

oe = OrdinalEncoder()
df["BP"] = oe.fit_transform(df[["BP"]])
df["Sex"] = oe.fit_transform(df[["Sex"]])
df["Cholesterol"] = oe.fit_transform(df[["Cholesterol"]])
df["Drug"] = oe.fit_transform(df[["Drug"]])

print(df)

x = df.iloc[:, 0:-1]
y = df.iloc[:, -1]
print(x)
print(y)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    x, y, random_state=42, test_size=0.2
)

print(X_train)
print(y_train)


from sklearn.tree import DecisionTreeClassifier

clf_gini = DecisionTreeClassifier(criterion="gini", random_state=0)
clf_gini.fit(X_train, y_train)
y_pred_gini = clf_gini.predict(X_test)
print(y_pred_gini)


from sklearn.metrics import accuracy_score

print(accuracy_score(y_pred_gini, y_test))

from sklearn import tree

plt.figure(figsize=(10, 10))
tree.plot_tree(clf_gini.fit(X_train, y_train))
plt.show()


clf_entropy = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=0)
clf_entropy.fit(X_train, y_train)
y_pred_entropy = clf_entropy.predict(X_test)

print(accuracy_score(y_test, y_pred_entropy))
plt.figure(figsize=(10, 10))
tree.plot_tree(clf_entropy.fit(X_train, y_train))
plt.show()
