import pandas as pd

a = pd.read_csv("Scikit-Learn/flawer_csv file/flower2.csv")
print(a)


from sklearn.feature_selection import SequentialFeatureSelector

x = a.iloc[:, :-1]
y = a["species"]
print(x.shape)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
sf = SequentialFeatureSelector(lr, n_features_to_select=3)
# sf = SequentialFeatureSelector(lr, n_features_to_select=3, direction="forward")

sf.fit(x, y)

print(sf.feature_names_in_)
