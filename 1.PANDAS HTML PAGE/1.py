import pandas as pd

df = pd.read_csv("Scikit-Learn/bb.csv")
print(df)

from pandas_profiling import ProfileReport

prof = ProfileReport(df)
prof.to_file(output_file="output.html")
