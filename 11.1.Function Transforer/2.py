import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

a = pd.read_csv("Scikit-Learn\\ff.csv")
print(a)

print(a.isnull().sum())

sns.distplot(a["area"])
plt.show()

q1 = a["area"].quantile(0.25)
q2 = a["area"].quantile(0.75)
IQR = q2 - q1

min_range = q1 - (1.5 * IQR)
max_range = q2 + (1.5 * IQR)
print(min_range, max_range)
new_data = a[a["area"] <= max_range]
print(a.shape)
print(new_data.shape)

sns.distplot(new_data["area"])
plt.show()

from sklearn.preprocessing import FunctionTransformer

# ft = FunctionTransformer(func=lambda x: x**2)
ft = FunctionTransformer(func=np.log1p)

ft.fit(a[["area"]])
a["area_ft"] = ft.transform(a[["area"]])

plt.subplot(1, 2, 1)
sns.distplot(a["area"])
plt.title("Befor")
plt.show()


plt.subplot(1, 2, 2)
sns.distplot(a["area_ft"])
plt.title("After")
plt.show()
