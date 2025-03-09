import pandas as pd

a = pd.read_csv("Scikit-Learn/cell_samples.csv")
print(a)

print(a.isnull().sum())
print(a.info())

print(a["Class"].value_counts())
a.drop(columns="BareNuc", inplace=True)

x = a.iloc[:, :-1]
y = a["Class"]


from imblearn.under_sampling import RandomUnderSampler

ru = RandomUnderSampler()
ru_x, ru_y = ru.fit_resample(x, y)

print(ru_y.value_counts())

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    ru_x, ru_y, test_size=0.2, random_state=42
)


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(x_train, y_train)
print(lr.score(x_test, y_test) * 100)
