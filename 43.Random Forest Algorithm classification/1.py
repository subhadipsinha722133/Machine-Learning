import pandas as pd

df = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
)

print(df)

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

col_names = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]

df.columns = col_names
print(df)

print(df.info())
print(df.describe(include="all"))
print(df.describe(include="all").T)


for col in col_names:
    print(df[col].value_counts())


print(df.duplicated())
print(df[df.duplicated()])


from sklearn.preprocessing import OrdinalEncoder

oe = OrdinalEncoder()
df["buying"] = oe.fit_transform(df[["buying"]])
df["maint"] = oe.fit_transform(df[["maint"]])
df["doors"] = oe.fit_transform(df[["doors"]])
df["persons"] = oe.fit_transform(df[["persons"]])
df["lug_boot"] = oe.fit_transform(df[["lug_boot"]])
df["safety"] = oe.fit_transform(df[["safety"]])
df["class"] = oe.fit_transform(df[["class"]])
print(df.head())


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

from sklearn.ensemble import RandomForestClassifier

clf1 = RandomForestClassifier()
print(clf1.fit(X_train, y_train))

pred1 = clf1.predict(X_test)
print(pred1)


from sklearn.metrics import accuracy_score

print(accuracy_score(pred1, y_test))


from yellowbrick.model_selection import validation_curve

num_est = [100, 200, 450, 500, 750, 1000]
print(
    validation_curve(
        RandomForestClassifier(),
        X=X_train,
        y=y_train,
        param_name="max_depth",
        param_range=num_est,
        scoring="accuracy",
        cv=3,
    )
)

clf2 = RandomForestClassifier(
    n_estimators=1000, min_samples_split=3, max_depth=15, random_state=0
)

clf2.fit(X_train, y_train)

pred2 = clf2.predict(X_test)
print(accuracy_score(pred2, y_test))

feature_scores = pd.Series(
    clf2.feature_importances_, index=X_train.columns
).sort_values(ascending=False)
print(feature_scores)

sns.barplot(x=feature_scores, y=feature_scores.index)
plt.xlabel("Feature Importance")
plt.show()


clf3 = RandomForestClassifier()
Xn = df.drop(["doors", "lug_boot", "maint"], axis=1)
yn = df["class"]
X_trainn, X_testn, y_trainn, y_testn = train_test_split(Xn, yn, test_size=0.3)
clf3.fit(X_trainn, y_trainn)
new_pred = clf3.predict(X_testn)
print(accuracy_score(new_pred, y_testn))
