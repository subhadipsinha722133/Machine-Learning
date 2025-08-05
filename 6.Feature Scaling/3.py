import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

d = pd.read_csv("Scikit-Learn\CO2_emission.csv")
print(d)

print(d.isnull().sum())
print(d.describe())

sns.distplot(d["CO2_Emissions"])
plt.show()

from sklearn.preprocessing import MinMaxScaler

ms = MinMaxScaler()
ms.fit(d[["CO2_Emissions"]])
print(ms.transform(d[["CO2_Emissions"]]))
d["CO2_Emissions_minmax"] = ms.transform(d[["CO2_Emissions"]])
print(d.head())
