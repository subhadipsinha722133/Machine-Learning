import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("Scikit-Learn\\null_.csv")
print(df.head())
print(df.shape)


missing_value_clm_gre_20 = ["Alley", "FireplaceQu", "PoolQC", "Fence", "MiscFeature"]
df2_drop_clm = df.drop(columns=missing_value_clm_gre_20)
print(df2_drop_clm.shape)


df3_num = df2_drop_clm.select_dtypes(include=["int64", "float64"])
print(df3_num.shape)

print(df3_num.isnull().sum())


num_var_miss = ["LotFrontage", "MasVnrArea", "GarageYrBlt"]
print(df3_num[num_var_miss][df3_num[num_var_miss].isnull().any(axis=1)])

print("===================================")
print(df["LotConfig"].unique())
print(df[df.loc[:, "LotConfig"] == "Inside"])

print(
    df[df.loc[:, "LotConfig"] == "Inside"]["LotFrontage"].replace(
        np.nan, df[df.loc[:, "LotConfig"] == "Inside"]["LotFrontage"].mean()
    )
)

print()
df_copy = df.copy()
for var__class in df["LotConfig"].unique():
    (
        df_copy.update(
            df[df.loc[:, "LotConfig"] == "Inside"]["LotFrontage"].replace(
                np.nan, df[df.loc[:, "LotConfig"] == "Inside"]["LotFrontage"].mean()
            )
        )
    )

print(df_copy.isnull().sum())


print(";;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;")
df_copy = df.copy()
num_vars_miss = ["LotFrontage", "MasVnrArea", "GarageYrBlt"]
cat_vars = ["LotConfig", "MasVnrType", "GarageType"]
for cat_var, num_var_miss in zip(cat_vars, num_vars_miss):
    for var_class in df[cat_var].unique():
        df_copy.update(
            df[df.loc[:, cat_var] == var_class][num_var_miss].replace(
                np.nan, df[df.loc[:, cat_var] == var_class][num_var_miss].mean()
            )
        )
print(df_copy[num_var_miss].isnull().sum())


print(df_copy[df_copy[["MasVnrType"]].isnull().any(axis=1)])
print()
print(df_copy[df_copy[["GarageType"]].isnull().any(axis=1)])


print(
    "[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]"
)


df_copy = df.copy()
num_vars_miss = ["LotFrontage", "MasVnrArea", "GarageYrBlt"]
cat_vars = ["LotConfig", "Exterior2nd", "KitchenQual"]
for cat_var, num_var_miss in zip(cat_vars, num_vars_miss):
    for var_class in df[cat_var].unique():
        df_copy.update(
            df[df.loc[:, cat_var] == var_class][num_var_miss].replace(
                np.nan, df[df.loc[:, cat_var] == var_class][num_var_miss].mean()
            )
        )
print(df_copy[num_vars_miss].isnull().sum())

plt.figure(figsize=(10, 10))
sns.set()
for i, var in enumerate(num_vars_miss):
    plt.subplot(2, 2, i + 1)
    sns.distplot(
        df[var],
        bins=20,
        kde_kws={"linewidth": 8, "color": "red"},
        label="Original",
    )
    sns.distplot(
        df_copy[var],
        bins=20,
        kde_kws={"linewidth": 5, "color": "green"},
        label="Mean",
    )
    plt.legend()

# Mean
df_copy_median = df.copy()
num_vars_miss = ["LotFrontage", "MasVnrArea", "GarageYrBlt"]
cat_vars = ["LotConfig", "Exterior2nd", "KitchenQual"]
for cat_var, num_var_miss in zip(cat_vars, num_vars_miss):
    for var_class in df[cat_var].unique():
        df_copy_median.update(
            df[df.loc[:, cat_var] == var_class][num_var_miss].replace(
                np.nan, df[df.loc[:, cat_var] == var_class][num_var_miss].median()
            )
        )

print(df_copy_median[num_vars_miss].isnull().sum())
