import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

a = pd.read_csv("Scikit-Learn\PDB_Load_History.csv")
print(a)
# print(a.isnull().sum())
print(a.info())

sns.distplot(a["demand"])
plt.show()

print(a.describe())
a = a.iloc[:, 1:]
print(a)

from sklearn.preprocessing import StandardScaler

at = StandardScaler()
a = pd.DataFrame(at.fit_transform(a), columns=a.columns)
print(a)
