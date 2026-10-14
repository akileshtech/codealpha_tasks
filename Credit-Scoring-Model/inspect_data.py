import pandas as pd
data = pd.read_csv(
    "german.data",
    sep=r"\s+",
    header=None,
    engine="python"
    )
print("Data shape:",data.shape)

print("\nFirst 5 rows:")
print(data.head())

print("\nColumn names:")
print(data.columns)

print("\nData types:")
print(data.dtypes)

print("\nTarget column values:")
print(data[20].value_counts())