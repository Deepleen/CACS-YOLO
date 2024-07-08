from ultralytics import YOLO

# Load CACS-YOLO
model = YOLO(model='./ultralytics/cfg/models/CACS-YOLO.yaml')
# Load YOLOv8m
# model = YOLO(model='./ultralytics/cfg/models/YOLOv8m.yaml')
# Load YOLOv8m-CAM
# model = YOLO(model='./ultralytics/cfg/models/YOLOv8m-CAM.yaml')
# Load YOLOv8m-CSO
# model = YOLO(model='./ultralytics/cfg/models/YOLOv8m-CSO.yaml')

# train by UPID
model.train(data='./ultralytics/cfg/datasets/UPID.yaml', device=0, epochs=500)
# train by SFID
# model.train(data='./ultralytics/cfg/datasets/SFID.yaml', device=0, epochs=500)
# train by SRID
# model.train(data='./ultralytics/cfg/datasets/SRID.yaml', device=0, epochs=500)
# train by SFRID
# model.train(data='./ultralytics/cfg/datasets/SFRID.yaml', device=0, epochs=600)
