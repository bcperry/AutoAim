# AutoAim

The Halo Auto Aim Bot is a python file consisting of methods for screen capture, object
detection inference, and mouse movement.

Within a loop, we first capture the screen on which Halo is being played. From here, the
image is sent into the YOLOv5[3] model for inference. The output of this inference is a
list of bounding boxes with enemy detections. From here, we loop through all of the
predictions to find the object closest to the current aim point. Once this is determined,
we calculate the relative distance to the target, apply a scaling factor to reduce the
movement speed and mitigate overshooting the target, and command mouse
movement.

Model Selection:

Due to the fast paced nature of the game, any model that we select needs to be able to
return accurate results quickly during dynamic gameplay. Additionally, since the game
can be graphically demanding, we need to select a model that does not use too much of
the GPU’s resources. Three models were selected and tested.

● You Only Look Once (YOLO) v1[4]:
YOLOv1 is a single shot object detection model which is computationally light,
but has a relatively low precision and recall compared to other state of the art
object detection models.

● Faster RCNN[5]:
FRCNN is a two stage object detection model which is also relatively
computationally light, though more so than YOLOv1. This model is capable of
higher precision and recall than single stage models, but is slower in doing so.

● YOLOv5[3]:
YOLOv5 is the latest version in the YOLO model family. It is also a single stage
detector, but is capable of higher precision and recall than YOLOv1, while at the
same time maintaining its ability to perform very fast inference.


# Auto Aim for Halo Infinite using YOLOv5
---
## Blaine Perry
---
# to run model using Docker
# build the container
docker build -t autoaim .
# run the container, attaching the local directory to the user folder in the container
docker run --rm -d -itp 8888:8888 -v %cd%:/app --gpus all autoaim