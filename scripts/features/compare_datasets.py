import pandas as pd

df = pd.read_csv("outputs/standardized/standardized_motion_taxonomy.csv")

print("\n🔥 Média de movimento por dataset:\n")
print(df.groupby("dataset")["motion_signal_normalized"].describe())

print("\n🔥 Distribuição de intensidade por dataset:\n")
print(pd.crosstab(df["dataset"], df["behavior_intensity"]))

print("\n🔥 Tipo de comportamento por dataset:\n")
print(pd.crosstab(df["dataset"], df["behavior_type"]))