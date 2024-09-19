import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

train = pd.read_csv("Scikit-Learn\\train2.csv")
test = pd.read_csv("Scikit-Learn\\test2.csv")

print("shape of train df = ", train.shape)
print("shape of test df = ", test.shape)
print()
X_train = train.drop(columns="SalePrice", axis=1)
y_train = train["SalePrice"]
X_test = test.copy()
print("Shape of X_train =", X_train.shape)
print("Shape of y_train =", y_train.shape)
print("Shape of X_test =", X_test.shape)

print()
# Missing value imputation

isnull_sum = X_train.isnull().sum()
print(isnull_sum)

print()
# finding the numerical variable which have mising value
num_vars = X_train.select_dtypes(include=["int64", "float64"]).columns
num_vars_miss = [var for var in num_vars if isnull_sum[var] > 0]
print(num_vars_miss)

print()

# finding the categorical variable which have mising value
cat_vars = X_train.select_dtypes(include=["O"]).columns
cat_vars_miss = [var for var in cat_vars if isnull_sum[var] > 0]
print(cat_vars_miss)


print(
    "======================================================================================"
)
num_var_mean = ["LotFrontage"]
num_var_median = ["MasVnrArea", "GarageYrBlt"]
cat_vars_mode = [
    "Alley",
    "MasVnrType",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "Electrical",
    "FireplaceQu",
]
cat_vars_missing = [
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "PoolQC",
    "Fence",
    "MiscFeature",
]

num_var_mean_imputer = Pipeline(steps=[("imputer", SimpleImputer(strategy="mean"))])
num_var_median_imputer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
cat_vars_mode_imputer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="most_frequent"))]
)
cat_vars_missing_imputer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="constant", fill_value="missing"))]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("mean_imputer", num_var_mean_imputer, num_var_mean),
        ("median_imputer", num_var_median_imputer, num_var_median),
        ("mode_imputer", cat_vars_mode_imputer, cat_vars_mode),
        ("missing_imputer", cat_vars_missing_imputer, cat_vars_missing),
    ]
)
print(preprocessor.fit(X_train))
print()
print(preprocessor.transform)
print()
print(
    preprocessor.named_transformers_["mean_imputer"].named_steps["imputer"].statistics_
)

print()

print(train["LotFrontage"].mean())

print()
print(
    preprocessor.named_transformers_["mode_imputer"].named_steps["imputer"].statistics_
)
print()

X_train_clean = preprocessor.transform(X_train)
X_test_clean = preprocessor.transform(X_test)
print(X_train_clean)
print()

print(preprocessor.transformers_)

X_train_clean_miss_var = pd.DataFrame(
    X_train_clean,
    columns=num_var_mean + num_var_median + cat_vars_mode + cat_vars_missing,
)
print(X_train_clean_miss_var.head())

print()
print(X_train_clean_miss_var.isnull().sum().sum())

print()
print(train["Alley"].value_counts())

print()
print(X_train_clean_miss_var["Alley"].value_counts())
print()
print(X_train_clean_miss_var["MiscFeature"].value_counts())
