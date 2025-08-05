import pandas as pd

a = pd.read_csv("Scikit-Learn/test.csv")
print(a)
print(a.describe())

import matplotlib.pyplot as plt
import seaborn as sns

sns.boxplot(a["Fare"])
plt.show()

q1 = a["Fare"].quantile(0.25)
q2 = a["Fare"].quantile(0.75)
IQR = q2 - q1
min_range = q1 - (1.5 * IQR)
max_range = q2 + (1.5 * IQR)
aa = a[a["Fare"] <= max_range]

sns.boxplot(aa["Fare"])
plt.show()
print(a.shape)
print(aa.shape)
