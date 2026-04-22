import cv2
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm


def compute_optical_flow(prev_gray, curr_gray):
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        curr_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return magnitude, angle


def extract_sequence_features(sequence_path: Path):
    frame_files = sorted(sequence_path.glob("*.jpg"))

    if len(frame_files) < 2:
        return None

    prev = cv2.imread(str(frame_files[0]))
    if prev is None:
        return None

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    mean_magnitudes = []
    max_magnitudes = []

    for frame_path in frame_files[1:]:
        curr = cv2.imread(str(frame_path))
        if curr is None:
            continue

        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        magnitude, _ = compute_optical_flow(prev_gray, curr_gray)

        mean_magnitudes.append(float(np.mean(magnitude)))
        max_magnitudes.append(float(np.max(magnitude)))

        prev_gray = curr_gray

    if len(mean_magnitudes) == 0:
        return None

    mean_magnitudes = np.array(mean_magnitudes)
    max_magnitudes = np.array(max_magnitudes)

    threshold = np.mean(mean_magnitudes)

    return {
        "motion_mean": float(np.mean(mean_magnitudes)),
        "motion_std": float(np.std(mean_magnitudes)),
        "motion_max": float(np.max(max_magnitudes)),
        "high_motion_frames": int(np.sum(mean_magnitudes > threshold)),
        "pairs_processed": int(len(mean_magnitudes)),
    }


def main():
    parser = argparse.ArgumentParser(description="Extração de sinais de movimento com optical flow")
    parser.add_argument("--input_dir", type=str, required=True, help="Diretório com as sequências de frames")
    parser.add_argument("--output_path", type=str, required=True, help="Caminho do CSV de saída")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output_path)

    if not input_dir.exists():
        raise FileNotFoundError(f"Pasta não encontrada: {input_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for sequence in tqdm(sorted(input_dir.iterdir()), desc="Extracting motion features"):
        if not sequence.is_dir():
            continue

        features = extract_sequence_features(sequence)
        if features is None:
            continue

        features["sequence"] = sequence.name
        rows.append(features)

    if len(rows) == 0:
        print("Nenhuma sequência processada.")
        return

    df = pd.DataFrame(rows)
    df = df[[
        "sequence",
        "motion_mean",
        "motion_std",
        "motion_max",
        "high_motion_frames",
        "pairs_processed"
    ]]
    df.to_csv(output_path, index=False)

    print(f"Motion features extraídas com sucesso: {output_path}")


if __name__ == "__main__":
    main()