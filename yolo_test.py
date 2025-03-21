from ultralytics import YOLO
import matplotlib.pyplot as plt

import cv2

img_path = "datasets/erick/val/images/7166_random_7166_accept_7166_random_2ND.jpg"
img = cv2.imread(img_path)
print(img.shape)

# model = YOLO("runs/detect/train9/weights/best.pt")  # Using nano model from Ultralytics
model = YOLO("C://Users/rocha/Documents/NACRE/slots/yolov8n_custom3/weights/best.pt")  # Using nano model from Ultralytics

print(model.names)
model.names = {0: 'Slots'}

# Perform object detection on an image
results = model(img_path)
results[0].show(line_width=2)

img = cv2.resize(img, (640, 640))
results = model(img)
results[0].show(line_width=1)
