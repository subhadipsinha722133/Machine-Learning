import sklearn
import pandas as pd
from sklearn.datasets import fetch_openml

df = fetch_openml("titanic", version=1, as_frame=True)["data"]
print(df.info())

print(df.isnull())

print(df.isnull().sum())


import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
miss_value_per = pd.DataFrame((df.isnull().sum() / len(df)) * 100)
miss_value_per.plot(
    kind="bar", title="Missing value in percentage", ylabel="percentage"
)
plt.show()
