import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# df = pd.read_csv("Scikit-Learn\\aa.csv")
# df = pd.read_csv("Scikit-Learn\\mm.csv")
df = pd.read_csv("Scikit-Learn\\null_.csv")
# df = pd.read_csv("Scikit-Learn\\student.csv")


print(df.head())

print(df.shape)

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
print(df.head())
print
print(df.info())
print(df.isnull())
# print(df.isnull().sum())

# plt.figure(figsize=(25, 25))
sns.heatmap(df.isnull())
plt.show()

print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")
null_var = df.isnull().sum() / df.shape[0] * 100
print(null_var)
print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")

drop_column = null_var[null_var > 17].keys()
print(drop_column)
aa_drop_colom = df.drop(columns=drop_column)
print(drop_column.shape)
sns.heatmap(aa_drop_colom.isnull())
plt.show()
print("============================================================")


bb_drop_row = aa_drop_colom.dropna()
print(bb_drop_row)
print()
sns.heatmap(bb_drop_row.isnull())
plt.show()

print(
    "=====================\\\\\\\\\\\\\\\\\\\\\\\======================================="
)

print(bb_drop_row.isnull())
print(bb_drop_row.isnull().sum())

print(bb_drop_row.select_dtypes(include=["int64", "Float64"]).columns)

sns.displot(df["MSSubClass"])
plt.show()


num_var = [
    "BsmtUnfSF",
    "TotalBsmtSF",
    "1stFlrSF",
    "2ndFlrSF",
    "LowQualFinSF",
    "GrLivArea",
    "BsmtFullBath",
    "BsmtHalfBath",
    "FullBath",
    "HalfBath",
    "BedroomAbvGr",
    "KitchenAbvGr",
    "TotRmsAbvGrd",
    "Fireplaces",
    "GarageYrBlt",
    "GarageCars",
    "GarageArea",
    "WoodDeckSF",
    "OpenPorchSF",
    "EnclosedPorch",
    "3SsnPorch",
    "ScreenPorch",
    "PoolArea",
    "MiscVal",
    "MoSold",
    "YrSold",
    "SalePrice",
]


print(df["OverallCond"].value_counts())

print(df["OverallCond"].value_counts() / bb_drop_row.shape[0] * 100)

print(
    pd.concat(
        [
            df["MSZoning"].value_counts() / bb_drop_row.shape[0] * 100,
            bb_drop_row["MSZoning"].value_counts() / bb_drop_row.shape[0] * 100,
        ],
        axis=1,
        keys=["MSZoning_org", "MSZoning_clean"],
    )
)
print("////////////////////////////////////////////////////")


def cat_var_dist(var):
    return pd.concat(
        [
            df[var].value_counts() / df.shape[0] * 100,
            bb_drop_row[var].value_counts() / bb_drop_row.shape[0] * 100,
        ],
        axis=1,
    )


print(cat_var_dist("MSZoning"))
