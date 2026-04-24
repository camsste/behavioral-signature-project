from pathlib import Path
import shutil


def restructure(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    for class_dir in input_dir.iterdir():
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name.lower()

        for file in list(class_dir.glob("*.png")) + list(class_dir.glob("*.PNG")):
            parts = file.stem.split("_")

            if len(parts) < 3:
                continue

            video_name = "_".join(parts[:2])
            frame_number = parts[-1]

            target_dir = output_dir / class_name / video_name
            target_dir.mkdir(parents=True, exist_ok=True)

            new_name = f"frame_{int(frame_number):06d}.png"
            shutil.copy2(file, target_dir / new_name)

    print("UCF reorganizado com sucesso.")


if __name__ == "__main__":
    restructure(
        input_dir="/home/camile-stefany/Downloads/archive(1)/Test",
        output_dir="datasets/ucf/raw"
    )