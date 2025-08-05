import pandas as pd


a = pd.read_csv("Scikit-Learn\cust_segmentation_Data.csv")
print(a)

input_data = a.iloc[:, :-1]
output_data = a["DebtIncomeRatio"]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    input_data, output_data, test_size=0.25
)
print(x_train)

print(y_train)

print(a.shape)
print(x_train.shape)
