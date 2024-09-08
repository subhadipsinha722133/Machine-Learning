import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
names = iris.target_names

# Check the shape of features and target
print(X.shape)
print(y.shape)

# Create a DataFrame for easier visualization and analysis
df = pd.DataFrame(X, columns=iris.feature_names)
df["species"] = iris.target
df["species"] = df["species"].replace(
    to_replace=[0, 1, 2], value=["setosa", "versicolor", "virginica"]
)
print(df)

# Visualize the dataset with pairplot
import seaborn as sns

sns.pairplot(data=df, hue="species", palette="Set2")
plt.show()

# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=1
)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# Train the SVM model
from sklearn.svm import SVC

svm = SVC(kernel="linear", random_state=0)
svm.fit(X_train, y_train)

# Make predictions on the test set (corrected)
pred = svm.predict(X_test)
print(pred)

# Evaluate the model
from sklearn.metrics import accuracy_score, confusion_matrix

print("Accuracy:", accuracy_score(y_test, pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))

rbf_svm = SVC(kernel="rbf", random_state=0)
rbf_svm.fit(X_train, y_train)
rbf_pred = rbf_svm.predict(X_test)

print(accuracy_score(y_test, rbf_pred))
