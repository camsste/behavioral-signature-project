import pandas as pd

df = pd.read_csv("outputs/standardized/standardized_motion_taxonomy.csv")

# filtrar só UCF
ucf = df[df["dataset"] == "ucf"]

# ordenar por intensidade
ucf_sorted = ucf.sort_values(by="motion_signal_normalized", ascending=False)

print("\n🔥 Ranking de intensidade (UCF):\n")
print(ucf_sorted[[
    "class",
    "behavior_type",
    "behavior_intensity",
    "motion_signal_normalized"
]])

# agrupamento por tipo
print("\n🔥 Média de intensidade por tipo de comportamento:\n")
print(ucf.groupby("behavior_type")["motion_signal_normalized"].mean().sort_values(ascending=False))

# agrupamento por intensidade
print("\n🔥 Classes por nível de intensidade:\n")
print(pd.crosstab(ucf["behavior_intensity"], ucf["class"]))