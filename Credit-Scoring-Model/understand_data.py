import pandas as pd
data = pd.read_csv(
    "german.data",
    sep=r"\s+",
    header=None,
    engine="python"
    )
print("Rows:", data.shape[0])
print("columns:", data.shape[1])

print("\nFirst customer:")
print(data.iloc[0])

print("\nTarget column values:")
print(data[20].value_counts())

print("\nTargets percentages:")
print(data[20].value_counts(normalize=True) *100)
