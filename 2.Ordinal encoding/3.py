import pandas as pd

a = pd.read_csv("Scikit-Learn\\cc.csv")
print(a)
import matplotlib.pyplot as plt
import seaborn as sns

sns.pairplot(data=a)
plt.show()

print(a.info())

b = a.select_dtypes(include="object")
print(b.info())
from sklearn.preprocessing import OrdinalEncoder

le = OrdinalEncoder()
b["TotalCharges"] = le.fit_transform(b[["TotalCharges"]])
b["PaymentMethod"] = le.fit_transform(b[["PaymentMethod"]])
b["PaperlessBilling"] = le.fit_transform(b[["PaperlessBilling"]])
print(b)
