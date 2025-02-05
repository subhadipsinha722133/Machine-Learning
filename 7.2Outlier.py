import pandas as pd

a = pd.read_csv("Scikit-Learn\CO2_emission.csv")
print(a)

print(a.info())

import seaborn as sns
import matplotlib.pyplot as plt

print(a.describe())
sns.boxplot(x="CO2_Emissions", data=a)
plt.show()

min_range = a["CO2_Emissions"].mean() - (3 * a["CO2_Emissions"].std())
max_range = a["CO2_Emissions"].mean() + (3 * a["CO2_Emissions"].std())
print(min_range, max_range)
new_data = a[a["CO2_Emissions"] <= max_range]

sns.boxplot(x="CO2_Emissions", data=a)
plt.show()

z_score = (a["CO2_Emissions"] - a["CO2_Emissions"].mean()) / (a["CO2_Emissions"].std())
print(z_score)
