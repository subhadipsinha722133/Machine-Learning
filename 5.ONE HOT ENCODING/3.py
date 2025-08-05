import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Load the dataset, ensuring the correct file path
tips_df = pd.read_csv("Scikit-Learn\\5.ONE HOT ENCODING\\tips.csv")
print(tips_df)

# One-hot encoding with pd.get_dummies()
dummy_df = pd.get_dummies(tips_df)
print(dummy_df)

# Using drop_first=True to remove first category for each categorical feature
print("..........................")
print(pd.get_dummies(tips_df, drop_first=True))
print("............................................................................")
# One-hot encoding using sklearn's OneHotEncoder
oh_enc = OneHotEncoder(
    sparse_output=False
)  # Setting sparse_output=False to get a dense array
oh_enc_arr = oh_enc.fit_transform(tips_df[["sex", "smoker", "day", "time"]])
print(oh_enc_arr)
print(dummy_df.keys())
# Get the one-hot encoded column names automatically
oh_enc_columns = oh_enc.get_feature_names_out(["sex", "smoker", "day", "time"])

# Convert the encoded array into a DataFrame
oh_enc_df = pd.DataFrame(oh_enc_arr, columns=oh_enc_columns)

print(oh_enc_df)
