import ultralytics
from ultralytics import YOLO

import torch

ultralytics.checks()

model = YOLO('yolov8s.pt') # pretrained model


##TRAIN 1
#model.train(data="C:/Users/chris/Documents/350_CV_Project/Final_data/config.yaml",epochs=5,patience=5,batch=8, lr0=0.0005,imgsz=640)

##TRAIN 2
#  https://docs.ultralytics.com/modes/train/#train-settings
model.train(data="C:/Users/chris/Documents/350_CV_Project/Final_data/config.yaml",epochs=15,patience=5,batch=16, lr0=0.0005,imgsz=640, single_cls=True)

metrics = model.val()  # no arguments needed, dataset and settings remembered
metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps   # a list contains map50-95 of each category
metrics.box.mp    # P
metrics.box.mr    # R

# model = YOLO('C:/Users/chris/Documents/350_CV_Project/runs/detect/train5/weights/best.pt')

# results = model(source='C:/Users/chris/Documents/350_CV_Project/Final_data/images/train', conf=0.5)  # Adjust conf parameter
# results.print()  # Or use results.save() or results.print() to output results


# import numpy as np
# from sklearn.metrics import precisionrecall_fscore_support

# thresholds = np.arange(0.1, 1.0, 0.1)
# best_threshold = 0.5
# best_f1_score = 0

# for thresh in thresholds:
#     results = model.predict(source='C:/Users/chris/Documents/350_CV_Project/Final_data/images/val', conf=thresh)