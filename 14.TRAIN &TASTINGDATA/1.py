# import pandas as pd
import seaborn as sns

# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df = sns.load_dataset("titanic")
print(df.head())

df2 = df[["survived", "pclass", "age", "parch"]]
print(df2.head())

df3 = df2.fillna(df2.mean())
print(df3)
X = df3.drop("survived", axis=1)
y = df3["survived"]
print("Shape of X = ", X.shape)
print("Shape of y = ", y.shape)
print()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
print("Shape of X_train = ", X_train.shape)
print("Shape of y_train = ", y_train.shape)
print("Shape of X_test = ", X_test.shape)
print("Shape of y_test = ", y_test.shape)


print()
print(X_train)
