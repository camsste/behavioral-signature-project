import pandas as pd
from pathlib import Path


def categorize_intensity(value, q1, q2):
    if value >= q2:
        return "alta_intensidade"
    elif value >= q1:
        return "media_intensidade"
    else:
        return "baixa_intensidade"


def map_behavior_type(class_name):
    class_name = class_name.lower()

    if class_name in ["fighting", "shooting", "assault"]:
        return "agressivo_violento"

    elif class_name in ["explosion", "arson", "roadaccidents"]:
        return "disruptivo_abrupto"

    elif class_name in ["robbery", "burglary", "stealing", "shoplifting"]:
        return "ilicito_furtivo"

    elif class_name in ["abuse", "arrest", "vandalism"]:
        return "conflito_intervencao"

    elif class_name in ["normalvideos"]:
        return "neutro"

    else:
        return "outro"


def main():
    input_path = Path("datasets/ucf/processed/motion_features.csv")
    output_dir = Path("datasets/ucf/processed/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    summary = (
        df.groupby("class")
        .agg(
            sequences=("sequence", "count"),
            motion_mean_avg=("motion_mean", "mean"),
            motion_mean_std=("motion_mean", "std"),
            motion_std_avg=("motion_std", "mean"),
            motion_max_avg=("motion_max", "mean"),
            motion_max_peak=("motion_max", "max"),
            high_motion_frames_avg=("high_motion_frames", "mean"),
            pairs_processed_avg=("pairs_processed", "mean"),
        )
        .reset_index()
    )

    summary["motion_signal_strength"] = (
        summary["motion_mean_avg"]
        + summary["motion_std_avg"]
        + summary["motion_max_avg"]
    )

    summary["motion_signal_normalized"] = (
        summary["motion_signal_strength"] / summary["pairs_processed_avg"]
    )

    q1 = summary["motion_signal_normalized"].quantile(0.33)
    q2 = summary["motion_signal_normalized"].quantile(0.66)

    summary["behavior_intensity"] = summary["motion_signal_normalized"].apply(
        lambda x: categorize_intensity(x, q1, q2)
    )

    summary["behavior_type"] = summary["class"].apply(map_behavior_type)

    summary = summary.sort_values(
        by="motion_signal_normalized",
        ascending=False
    )

    output_file = output_dir / "motion_summary_by_class.csv"
    summary.to_csv(output_file, index=False)

    print("\nTaxonomia comportamental interpretável:\n")
    print(summary[[
        "class",
        "motion_signal_normalized",
        "behavior_intensity",
        "behavior_type"
    ]])

    print(f"\nArquivo salvo em: {output_file}")


if __name__ == "__main__":
    main()