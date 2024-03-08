from ultralytics import YOLO
import os

# Load a model
model_dir = 'cfg'
scale = 'n'
model_name = f'yolov8{scale}-seg+ghost.yaml'
model = YOLO(f'{model_dir}/{model_name}')
# model = YOLO('yolov8n-seg.pt')  # load a pretrained model (recommended for training)

data_dir = "data"
data_name = "coco-seg.yaml"
# Train the model
results = model.train(data=os.path.join(data_dir, data_name), imgsz=640, batch=32,
                      epochs=100, device=0, patience=10, task="segment",
                      exist_ok=True, name="yolo+ghost", save_txt=True,)