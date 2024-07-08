from ultralytics import YOLO

# Load the best model
model = YOLO('./runs/detect/train1/weights/best.pt')

# Testing with UPID
model.val(data='./ultralytics/cfg/datasets/UPID.yaml', device=0)
# Testing with SFID
# model.val(data='./ultralytics/cfg/datasets/SFID.yaml', device=0)
# Testing with SRID
# model.val(data='./ultralytics/cfg/datasets/SRID.yaml', device=0)
# Testing with SFRID
# model.val(data='./ultralytics/cfg/datasets/SFRID.yaml', device=0)
