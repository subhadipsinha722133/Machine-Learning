import pandas as pd

a = pd.read_csv("Scikit-Learn/bb.csv")
print(a.isnull().sum())

import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(x="TV", y="Sales", data=a)
plt.show()

from sklearn.model_selection import train_test_split

X = a.iloc[:, :-1]
y = a["Sales"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(X_train)
