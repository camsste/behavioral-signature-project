import cv2
import os

input_dir = "datasets/raw"
output_dir = "datasets/processed"

os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(input_dir):
    if file.endswith(".mp4"):
        path = os.path.join(input_dir, file)
        cap = cv2.VideoCapture(path)

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % 30 == 0:
                out_path = os.path.join(output_dir, f"{file}_frame{frame_count}.jpg")
                cv2.imwrite(out_path, frame)

            frame_count += 1

        cap.release()

print("Pré-processamento básico concluído.")
