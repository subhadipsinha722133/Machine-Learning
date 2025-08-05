import pandas as pd

a = pd.read_csv("Scikit-Learn\\cc.csv")
print(a)
data = a["PaymentMethod"].unique()
print(data)

s = a["PaymentMethod"].fillna(a["PaymentMethod"].mode()[0])
print(s)
em = [
    [
        "Electronic check"
        "Mailed check"
        "Bank transfer (automatic)"
        "Credit card (automatic)"
    ]
]


from sklearn.preprocessing import OrdinalEncoder

le = OrdinalEncoder()
a["PaymentMethod"] = le.fit_transform(a[["PaymentMethod"]])
print(a)
