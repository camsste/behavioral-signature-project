import pandas as pd

df = pd.read_csv("datasets/shanghaitech/processed/analysis/motion_summary_shanghai.csv")

print("\n🔥 Dados Shanghai:\n")
print(df[[
    "sequence",
    "behavior_type",
    "behavior_intensity",
    "motion_signal_normalized",
    "anomaly_ratio"
]])

print("\n🔥 Média de movimento por tipo:\n")
print(df.groupby("behavior_type")["motion_signal_normalized"].mean())

print("\n🔥 Distribuição de intensidade:\n")
print(pd.crosstab(df["behavior_type"], df["behavior_intensity"]))