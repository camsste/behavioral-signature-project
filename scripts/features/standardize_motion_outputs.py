import pandas as pd
from pathlib import Path


def load_ucf():
    path = Path("datasets/ucf/processed/analysis/motion_summary_by_class.csv")
    df = pd.read_csv(path)

    df["dataset"] = "ucf"
    df["sequence"] = df["class"]
    df["motion_mean"] = df["motion_mean_avg"]
    df["motion_std"] = df["motion_std_avg"]
    df["motion_max"] = df["motion_max_avg"]
    df["pairs_processed"] = df["pairs_processed_avg"]

    return df[[
        "dataset",
        "class",
        "sequence",
        "behavior_type",
        "behavior_intensity",
        "motion_signal_normalized",
        "motion_mean",
        "motion_std",
        "motion_max",
        "pairs_processed"
    ]]


def load_shanghai():
    path = Path("datasets/shanghaitech/processed/analysis/motion_summary_shanghai.csv")
    df = pd.read_csv(path)

    df["dataset"] = "shanghaitech"
    df["class"] = df["behavior_type"]
    df["motion_mean"] = df["motion_mean"]
    df["motion_std"] = df["motion_std"]
    df["motion_max"] = df["motion_max"]

    return df[[
        "dataset",
        "class",
        "sequence",
        "behavior_type",
        "behavior_intensity",
        "motion_signal_normalized",
        "motion_mean",
        "motion_std",
        "motion_max",
        "pairs_processed"
    ]]


def main():
    output_dir = Path("outputs/standardized")
    output_dir.mkdir(parents=True, exist_ok=True)

    ucf = load_ucf()
    shanghai = load_shanghai()

    combined = pd.concat([ucf, shanghai], ignore_index=True)

    output_path = output_dir / "standardized_motion_taxonomy.csv"
    combined.to_csv(output_path, index=False)

    print("Arquivo padronizado salvo em:")
    print(output_path)

    print("\nPrévia:")
    print(combined.head(20))


if __name__ == "__main__":
    main()