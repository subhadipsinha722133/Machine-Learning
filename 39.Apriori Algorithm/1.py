import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

df = pd.read_csv("Scikit-Learn\\39.Apriori Algorithm\\Dataset.csv")
print(df.head())

# df["BROOKLYN"] = df["BROOKLYN"].str.split()

# basket = (
#     df[df["Chinese"] == "American"]
#     .groupby([" BROOKLYN", "BROOKLYN"])["z"]
#     .sum()
#     .unstack()
#     .reset_index()
#     .fillna(0)
#     .set_index("InvoiceNo")
# )

# print(basket)

import matplotlib.pyplot as plt

from apriori import apriori


df = pd.read_csv("Scikit-Learn\\Apriori Algorithm\\Dataset.csv", header=None)
print(df.head())

print(df.shape)
records = []

association_rules = apriori(
    min_support=0.0045, min_confidence=0.2, min_lift=3, min_length=2
)
association_results = list(
    association_rules
)  # convert the rules found by the apriori class into a li
