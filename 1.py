import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

a = pd.read_csv("Scikit-Learn/cancer_csv_file/survey lung cancer.csv")
print(a)

a = a.drop(columns="LUNG_CANCER")
print(a.isnull().sum())

n = a.select_dtypes(include="object")
m = pd.get_dummies(n, drop_first="True")

b = n.keys()
bb = m.keys()

from sklearn.preprocessing import OneHotEncoder

oe = OneHotEncoder(drop="first")
arr = oe.fit_transform(a[b]).toarray()
new = pd.DataFrame(arr, columns=[bb])
print(new)

a["GENDER"] = new["GENDER_M"]
print(a)

# sns.pairplot(data=a)
plt.show()

from sklearn.cluster import KMeans

k = []
for i in range(2, 21):
    km = KMeans(n_clusters=i, init="k-means++")
    km.fit(a)
    k.append(km.inertia_)

plt.plot([i for i in range(2, 21)], k, marker="o")
plt.xlabel("no of clusters")
plt.xticks([i for i in range(2, 21)])
plt.ylabel("k")
plt.grid(axis="x")
plt.show()

kmn = KMeans(n_clusters=2)
print(kmn.fit_predict(a))

a["predict"] = kmn.fit_predict(a)
print(a)


print(kmn.labels_)
from sklearn.metrics import silhouette_score

print(silhouette_score(a, labels=kmn.labels_))


ss = []
for i in range(2, 21):
    km1 = KMeans(n_clusters=i)
    km1.fit(a)
    ss.append(silhouette_score(a, km1.labels_))


plt.plot([j for j in range(2, 21)], ss)
plt.xticks([j for j in range(2, 21)])
plt.xlabel("no of clusters")
plt.ylabel("silhouette_score")
plt.grid(axis="x")
plt.show()
