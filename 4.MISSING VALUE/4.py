# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("Scikit-Learn\\null_.csv")

cat_vars = df.select_dtypes(include="object")
print(cat_vars.head())
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

print(cat_vars.isnull().sum())

print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
miss_val_per = cat_vars.isnull().mean() * 100
print(miss_val_per)


drop_vars = ["Alley", "FireplaceQu", "PoolQC", "Fence", "MiscFeature"]
cat_vars.drop(columns=drop_vars, axis=1, inplace=True)
print(cat_vars.shape)


print("[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]")

isnull_per = cat_vars.isnull().mean() * 100
miss_vars = isnull_per[isnull_per > 0].keys()
print(miss_vars)
print("||||||||||||||||||||||||||||||||||||")

print(cat_vars["MasVnrType"].fillna("Missing"))
print()
print(cat_vars["MasVnrType"].mode())

print()

print(cat_vars["MasVnrType"].value_counts())

print("//////////////////////")

print(cat_vars["MasVnrType"].fillna(cat_vars["MasVnrType"].mode()[0]))


print(cat_vars["MasVnrType"].fillna(cat_vars["MasVnrType"].mode()[0]).value_counts())
cat_vars_copy = cat_vars.copy()

for var in miss_vars:
    cat_vars_copy[var].fillna(cat_vars[var].mode()[0], inplace=True)
    print(var, "=", cat_vars[var].mode()[0])


print(cat_vars_copy.isnull().sum().sum())


plt.figure(figsize=(16, 9))
for i, var in enumerate(miss_vars):
    plt.subplot(4, 3, i + 1)
    plt.hist(cat_vars_copy[var], label="Impute")
    plt.hist(cat_vars[var].dropna(), label="Original")
plt.legend()
plt.show()


df.update(cat_vars_copy)
df.drop(columns=drop_vars, inplace=True)
print(df.select_dtypes(include="object").isnull().sum())
