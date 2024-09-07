import sklearn
from sklearn.datasets import load_iris

print(load_iris())
# print(load_iris(return_X_y=True))
X, y = load_iris(return_X_y=True)


from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X, y)
print(model.predict(X))
