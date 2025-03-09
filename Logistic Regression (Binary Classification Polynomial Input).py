import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

a = pd.read_csv("Scikit-Learn/cell_samples.csv")
print(a.isnull().sum())
print(a.info())
a.drop(columns="BareNuc", inplace=True)

sns.scatterplot(x="Clump", y="UnifSize", data=a, hue="Class")
plt.show()

x = a.iloc[:, :-1]
y = a["Class"]

from sklearn.preprocessing import PolynomialFeatures

pf = PolynomialFeatures(degree=4)
pf.fit(x)
x = pd.DataFrame(pf.transform(x))

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(x_train, y_train)
print(lr.score(x_test, y_test) * 100)
