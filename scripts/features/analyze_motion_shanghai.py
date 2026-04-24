import pandas as pd
from pathlib import Path


def categorize_intensity(value, q1, q2):
    if value >= q2:
        return "alta_intensidade"
    elif value >= q1:
        return "media_intensidade"
    else:
        return "baixa_intensidade"


def main():
    motion_path = Path("datasets/shanghaitech/processed/motion_features.csv")
    metadata_path = Path("datasets/shanghaitech/processed/metadata/metadata.csv")

    output_dir = Path("datasets/shanghaitech/processed/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    motion = pd.read_csv(motion_path)
    metadata = pd.read_csv(metadata_path)

    df = motion.merge(metadata, on="sequence", how="left")

    df["motion_signal_strength"] = (
        df["motion_mean"] + df["motion_std"] + df["motion_max"]
    )

    df["motion_signal_normalized"] = (
        df["motion_signal_strength"] / df["pairs_processed"]
    )

    q1 = df["motion_signal_normalized"].quantile(0.33)
    q2 = df["motion_signal_normalized"].quantile(0.66)

    df["behavior_intensity"] = df["motion_signal_normalized"].apply(
        lambda x: categorize_intensity(x, q1, q2)
    )

    df["behavior_type"] = df["anomaly_frames_total"].apply(
        lambda x: "anomalo" if x > 0 else "normal"
    )

    df = df.sort_values(by="motion_signal_normalized", ascending=False)

    output_file = output_dir / "motion_summary_shanghai.csv"
    df.to_csv(output_file, index=False)

    print("\nTaxonomia comportamental ShanghaiTech:\n")
    print(df[[
        "sequence",
        "motion_signal_normalized",
        "behavior_intensity",
        "behavior_type",
        "anomaly_ratio",
        "sampled_anomaly_ratio"
    ]])

    print(f"\nArquivo salvo em: {output_file}")


if __name__ == "__main__":
    main()