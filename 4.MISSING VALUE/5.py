import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

train = pd.read_csv("Scikit-Learn\\null_.csv")
test = pd.read_csv("Scikit-Learn\\test.csv")

print("shape of train df = ", train.shape)
print("shape of test df = ", test.shape)

print(train.head())


# X_train = train.drop(columns="Ticket")
# y_train = train["Ticket"]
# print("shape of X_train df = ", X_train.shape)
# print("shape of y_train df =", y_train.shape)

X_train = train.drop(columns="SalePrice")
y_train = train["SalePrice"]
print("shape of X_train df = ", X_train.shape)
print("shape of y_train df = ", y_train.shape)

print()

# X_train = train.drop(columns="Survived")
# y_train = train["Survived"]
# print("shape of X_train df = ", X_train.shape)
# print("shape of y_train df =", y_train.shape)

# Numerical Missing Value Imputation
num_vars = X_train.select_dtypes(include=["int64", "float64"]).columns

print(num_vars)

print(X_train[num_vars].isnull().sum())
print()

imputer_mean = SimpleImputer(strategy="mean")
print(imputer_mean.fit(X_train[num_vars]))

print(imputer_mean.statistics_)

print(imputer_mean.transform(X_train[num_vars]))

print("----------------------------------------------------")

print()


X_train[num_vars] = imputer_mean.transform(X_train[num_vars])
test[num_vars] = imputer_mean.transform(test[num_vars])
print(X_train[num_vars].isnull().sum())


# Categorical Missing Value Imputation

# train.drop("Name", axis=1, inplace=True)
# train.drop("Sex", axis=1, inplace=True)

# cat_vars = X_train.select_dtypes(include=["O"]).columns
# print(cat_vars)
# print(X_train[cat_vars].isnull().sum())


# imputer_mode = SimpleImputer(strategy="most_frequent")
# # # imputer_mean = SimpleImputer(strategy="constant", fill_value=99)
# print(imputer_mode)
# # print(imputer_mean)
# print(imputer_mean.fit(X_train[cat_vars]))
