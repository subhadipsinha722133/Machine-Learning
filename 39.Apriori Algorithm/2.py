import pandas as pd
import numpy as np

a = pd.read_csv("Scikit-Learn/39.Apriori Algorithm/Market_Basket_Optimisation.csv")
print(a.head())
print(a.info())
print(a.isnull().sum())
print(a.columns)
#
# all nul value are float
# collat all string and float values in deffrent variables

market = []
print(a.shape)
for i in range(0, a.shape[0]):
    cus = []
    for j in a.columns:
        # print(a[j][i])
        if type(a[j][i]) == str:
            cus.append(a[j][i])
    market.append(cus)

print(market)

l = []
for i in market:  # convart 1dimensional list
    for j in i:
        l.append(j)


import collections

# count the number of the elements
p = collections.Counter(l)

print(p.keys())
print(p.values())

d = pd.DataFrame({"Item_name": p.keys(), "value": p.values()})
print(d)
# pd.set_option("display.max_rows", 200)
print(d.sort_values(by=["value"], ascending=False))  # find highest sale value produck

from mlxtend.preprocessing.transactionencoder import TransactionEncoder

tr = TransactionEncoder()
tr.fit(market)

df = pd.DataFrame(tr.transform(market), columns=tr.columns_)
print(df)

from mlxtend.frequent_patterns import apriori

apri = apriori(df, min_support=0.07, use_colnames=True, max_len=3).sort_values(
    by=["support"]
)
print(apri)
