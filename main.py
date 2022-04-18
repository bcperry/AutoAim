import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', autoshape=False, classes=1)
print(model.se)

'''
YOLOmodel = torch.hub.load('ultralytics/yolov5', 'yolov5s', autoshape=False, classes=1)

# Image
img = 'https://ultralytics.com/images/zidane.jpg'

# Inference
results = YOLOmodel(img)

results.show()
'''