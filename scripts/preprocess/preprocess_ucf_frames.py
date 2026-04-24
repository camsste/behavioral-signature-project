from pathlib import Path
import pandas as pd


def process_ucf(input_dir, output_dir, max_per_class=3):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    metadata = []

    print(f"Input dir: {input_dir}")

    for class_dir in sorted(input_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        count = 0

        print(f"Classe: {class_name}")

        for sequence_dir in sorted(class_dir.iterdir()):
            if not sequence_dir.is_dir():
                continue

            frames = list(sequence_dir.glob("*.png")) + list(sequence_dir.glob("*.PNG"))

            if len(frames) == 0:
                continue

            metadata.append({
                "class": class_name,
                "sequence": sequence_dir.name,
                "frames_total": len(frames)
            })

            print(f"  Sequência: {sequence_dir.name} | Frames: {len(frames)}")

            count += 1

            if count >= max_per_class:
                break

    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(metadata)
    df.to_csv(output_dir / "metadata.csv", index=False)

    print(f"Total de sequências coletadas: {len(metadata)}")
    print("Metadata UCF gerado.")


if __name__ == "__main__":
    process_ucf(
        input_dir="datasets/ucf/raw",
        output_dir="datasets/ucf/processed/metadata",
        max_per_class=3
    )