import os
import shutil
import argparse
from pathlib import Path
import pandas as pd
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def list_sequence_dirs(frames_root: Path):
    return sorted([p for p in frames_root.iterdir() if p.is_dir()])


def list_frames(seq_dir: Path):
    return sorted(
        [p for p in seq_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    )


def main():
    parser = argparse.ArgumentParser(description="Preprocessamento de sequências de frames do ShanghaiTech")
    parser.add_argument("--input_dir", type=str, default="datasets/shanghaitech/raw")
    parser.add_argument("--output_dir", type=str, default="datasets/shanghaitech/processed")
    parser.add_argument("--stride", type=int, default=5, help="Seleciona 1 frame a cada N frames")
    parser.add_argument("--max_sequences", type=int, default=3, help="Número máximo de sequências para processar")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    frames_root = input_dir / "frames"
    labels_root = input_dir / "label"

    processed_frames_root = output_dir / "frames"
    metadata_root = output_dir / "metadata"

    processed_frames_root.mkdir(parents=True, exist_ok=True)
    metadata_root.mkdir(parents=True, exist_ok=True)

    if not frames_root.exists():
        raise FileNotFoundError(f"Pasta de frames não encontrada: {frames_root}")

    sequence_dirs = list_sequence_dirs(frames_root)
    if args.max_sequences:
        sequence_dirs = sequence_dirs[:args.max_sequences]

    metadata_rows = []

    for seq_dir in sequence_dirs:
        seq_name = seq_dir.name
        seq_out_dir = processed_frames_root / seq_name
        seq_out_dir.mkdir(parents=True, exist_ok=True)

        frames = list_frames(seq_dir)
        total_frames = len(frames)

        label_path = labels_root / f"{seq_name}.npy"
        label_exists = label_path.exists()

        anomaly_frames_total = None
        anomaly_ratio = None
        sampled_anomaly_frames = None
        sampled_anomaly_ratio = None

        labels = None
        if label_exists:
            labels = np.load(label_path)

            # garante vetor 1D
            labels = np.array(labels).reshape(-1)

            # se tamanho não casar, vamos usar o mínimo para evitar quebrar
            effective_len = min(len(labels), total_frames)

            anomaly_frames_total = int(np.sum(labels[:effective_len] > 0))
            anomaly_ratio = anomaly_frames_total / effective_len if effective_len > 0 else 0.0
        else:
            effective_len = total_frames

        if total_frames == 0:
            metadata_rows.append({
                "sequence": seq_name,
                "frames_total": 0,
                "frames_saved": 0,
                "stride": args.stride,
                "label_exists": label_exists,
                "anomaly_frames_total": anomaly_frames_total,
                "anomaly_ratio": anomaly_ratio,
                "sampled_anomaly_frames": sampled_anomaly_frames,
                "sampled_anomaly_ratio": sampled_anomaly_ratio,
            })
            continue

        saved_count = 0
        sampled_label_values = []

        for idx, frame_path in enumerate(frames[:effective_len]):
            if idx % args.stride == 0:
                out_name = f"frame_{saved_count:06d}{frame_path.suffix.lower()}"
                out_path = seq_out_dir / out_name
                shutil.copy2(frame_path, out_path)

                if labels is not None:
                    sampled_label_values.append(int(labels[idx] > 0))

                saved_count += 1

        if labels is not None and len(sampled_label_values) > 0:
            sampled_anomaly_frames = int(np.sum(sampled_label_values))
            sampled_anomaly_ratio = sampled_anomaly_frames / len(sampled_label_values)
        elif labels is not None:
            sampled_anomaly_frames = 0
            sampled_anomaly_ratio = 0.0

        metadata_rows.append({
            "sequence": seq_name,
            "frames_total": total_frames,
            "frames_saved": saved_count,
            "stride": args.stride,
            "label_exists": label_exists,
            "anomaly_frames_total": anomaly_frames_total,
            "anomaly_ratio": anomaly_ratio,
            "sampled_anomaly_frames": sampled_anomaly_frames,
            "sampled_anomaly_ratio": sampled_anomaly_ratio,
        })

    df = pd.DataFrame(metadata_rows)
    df.to_csv(metadata_root / "metadata.csv", index=False)

    print("Pipeline de frames concluído.")


if __name__ == "__main__":
    main()