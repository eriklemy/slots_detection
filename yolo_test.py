import cv2
from ultralytics import YOLO

img_path = r"erick/val/images/7039_14_151-6_2nd.jpg"
img = cv2.imread(img_path)
print(img.shape)

model = YOLO(r"slots\sahi\weights\best.pt") 

# Perform object detection on an image
results = model(img_path)
results[0].show(line_width=2)

img = cv2.resize(img, (640, 640))
results = model(img)
results[0].show(line_width=1)
