from ultralytics import YOLO

model = YOLO('yolov8l')

results = model.predict(
    source='input_videos/Spain_vs_France 2024.mp4',
    save=True,
    project='runs/detect',
    name='predict',
    exist_ok=True
)

print(results[0])
print('====================================================')
for box in results[0].boxes:
    print(box)