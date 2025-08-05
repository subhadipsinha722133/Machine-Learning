import pandas as pd

tips_df = pd.read_csv("Scikit-Learn\\5.ONE HOT ENCODING\\tips.csv")
print(tips_df)

dummy_df = pd.get_dummies(tips_df)
print(dummy_df)


print("..........................")
print(pd.get_dummies(tips_df, drop_first=True))


from sklearn.preprocessing import OneHotEncoder

oh_enc = OneHotEncoder(sparse_output=False)
oh_enc_arr = oh_enc.fit_transform(tips_df[["sex", "smoker", "day", "time"]])

# print(oh_enc_arr.toarray())
print(oh_enc_arr)

print(dummy_df.keys())

oh_enc_df = pd.DataFrame(
    oh_enc_arr,
    columns=[
        "sex_Female",
        "sex_Male",
        "smoker_No",
        "smoker_Yes",
        "day_Fri",
        "day_Sat",
        "day_Sun",
        "day_Thur",
        "time_Dinner",
        "time_Lunch",
    ],
)
print(oh_enc_df)
