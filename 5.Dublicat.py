import pandas as pd

data = {
    "name": ["a", "b", "a", "d", "e", "a"],
    "eng": [
        1,
        2,
        3,
        8,
        8,
        1,
    ],
    "data": [1, 2, 3, 4, 4, 1],
}
df = pd.DataFrame(data)
print(df)

print(df.duplicated())

# df["duplicated"] = df.duplicated()
# print(df)

print(df.drop_duplicates())
print()
