from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO("yolov8n.pt")

# Run inference on 'bus.jpg' with arguments
results = model.predict("/scratch/e1640a09/IROD/data/dataset/val/val_0.png", save=True, imgsz=320, conf=0.5, save_txt=True)




print(results)