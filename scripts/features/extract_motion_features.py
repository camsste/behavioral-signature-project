import cv2
import numpy as np
from pathlib import Path
import pandas as pd
import argparse


def compute_optical_flow(prev_frame, next_frame):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, next_gray,
        None,
        0.5, 3, 15, 3, 5, 1.2, 0
    )

    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return magnitude


def process_sequence(frame_paths):
    motions = []

    for i in range(len(frame_paths) - 1):
        f1 = cv2.imread(str(frame_paths[i]))
        f2 = cv2.imread(str(frame_paths[i + 1]))

        if f1 is None or f2 is None:
            continue

        mag = compute_optical_flow(f1, f2)
        motions.append(np.mean(mag))

    if len(motions) == 0:
        return None

    motions = np.array(motions)

    return {
        "motion_mean": float(np.mean(motions)),
        "motion_std": float(np.std(motions)),
        "motion_max": float(np.max(motions)),
        "high_motion_frames": int(np.sum(motions > np.mean(motions))),
        "pairs_processed": len(motions)
    }


def extract_features(input_dir, output_path):
    input_dir = Path(input_dir)

    results = []

    for class_dir in input_dir.iterdir():
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name

        for sequence_dir in class_dir.iterdir():
            if not sequence_dir.is_dir():
                continue

            frames = sorted(
                list(sequence_dir.glob("*.png")) +
                list(sequence_dir.glob("*.jpg"))
            )

            if len(frames) < 2:
                continue

            print(f"Processando: {class_name}/{sequence_dir.name}")

            features = process_sequence(frames)

            if features is None:
                continue

            features["class"] = class_name
            features["sequence"] = sequence_dir.name

            results.append(features)

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)

    print(f"\nTotal de sequências processadas: {len(results)}")
    print("Extração de motion concluída.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_path", required=True)

    args = parser.parse_args()

    extract_features(args.input_dir, args.output_path)