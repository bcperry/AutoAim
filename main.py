import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', autoshape=False, classes=1)
print(model.se)

'''
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', autoshape=False, classes=1)

# Image
img = 'https://ultralytics.com/images/zidane.jpg'

# Inference
results = model(img)

results.show()
'''