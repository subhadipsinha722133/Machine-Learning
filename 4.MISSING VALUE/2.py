import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("Scikit-Learn\\null_.csv")
print(df.head())
print(df.shape)
print(df.info())
print(df.isnull().sum())

print()
null_var = df.isnull().sum() / df.shape[0] * 100
print(null_var)


print("===========")
missing_value_clm_gre_20 = null_var[null_var > 20].keys()
print(missing_value_clm_gre_20)

df2_drop_clm = df.drop(columns=missing_value_clm_gre_20)
print(df2_drop_clm.shape)


print()
df3_num = df2_drop_clm.select_dtypes(include=["int64", "float64"])
print(df3_num.head())

# sns.heatmap(df3_num.isnull())
plt.show()
print("=====================================================")


print(df3_num[df3_num.isnull().any(axis=1)])

print(df3_num.isnull().sum())
missing_num_var = [var for var in df3_num.columns if df3_num[var].isnull().sum() > 0]
print(missing_num_var)


plt.figure(figsize=(10, 10))
sns.set()
for i, var in enumerate(missing_num_var):
    plt.subplot(2, 2, i + 1)
    sns.distplot(df3_num[var], bins=20, kde_kws={"linewidth": 5, "color": "#DC143C"})
# plt.show()

print("/////////////////////////////////")
df4_num_mean = df3_num.fillna(df3_num.mean())
print(df4_num_mean.isnull().sum().sum())


plt.figure(figsize=(10, 10))
sns.set()
for i, var in enumerate(missing_num_var):
    plt.subplot(2, 2, i + 1)
    sns.distplot(
        df3_num[var],
        bins=20,
        kde_kws={"linewidth": 8, "color": "red"},
        label="Original",
    )
    sns.distplot(
        df4_num_mean[var],
        bins=20,
        kde_kws={"linewidth": 5, "color": "green"},
        label="Mean",
    )
plt.legend()


df5_num_median = df3_num.fillna(df3_num.median())
print(df5_num_median.isnull().sum().sum())


plt.figure(figsize=(10, 10))
sns.set()
for i, var in enumerate(missing_num_var):
    plt.subplot(2, 2, i + 1)
    sns.distplot(
        df3_num[var],
        bins=20,
        hist=False,
        kde_kws={"linewidth": 8, "color": "red"},
        label="Original",
    )
    sns.distplot(
        df4_num_mean[var],
        bins=20,
        hist=False,
        kde_kws={"linewidth": 5, "color": "green"},
        label="Mean",
    )
    sns.distplot(
        df5_num_median[var],
        bins=20,
        hist=False,
        kde_kws={"linewidth": 3, "color": "k"},
        label="Median",
    )
    plt.legend()

for i, var in enumerate(missing_num_var):
    plt.figure(figsize=(10, 10))
    plt.subplot(3, 1, 1)
    sns.boxplot(df[var])
    plt.subplot(3, 1, 2)
    sns.boxplot(df4_num_mean[var])
    plt.subplot(3, 1, 3)
    sns.boxplot(df5_num_median[var])

df_concat = pd.concat(
    [
        df3_num[missing_num_var],
        df4_num_mean[missing_num_var],
        df5_num_median[missing_num_var],
    ],
    axis=1,
)
# pd.set_option("display.max_columns", None)
# pd.set_option("display.max_rows", None)
print(".....................................................................")
print(df_concat[df_concat.isnull().any(axis=1)].head())
