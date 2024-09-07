import pandas as pd
import numpy as np
from sklearn import datasets

data = datasets.load_wine()

X = data.data
y = data.target

df = pd.DataFrame(X, columns=data.feature_names)
df["wine Class"] = y

print(df)

print(df.isnull().sum())

print(df.describe())


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X = sc.fit_transform(X)

from sklearn.cluster import KMeans

wss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=0)
    kmeans.fit(X)
    wss.append(kmeans.inertia_)

import matplotlib.pyplot as plt

f3, ax = plt.subplots(figsize=(8, 6))
plt.plot(range(1, 11), wss)
plt.title("The Elbow Technique")
plt.xlabel("Number of Clusters")
plt.ylabel("Wss")
plt.show()


N = 3
k_means = KMeans(init="k-means++", n_clusters=N)
k_means.fit(X)
labels = k_means.labels_
print(labels)

from sklearn.metrics import accuracy_score

print(accuracy_score(labels, y))
