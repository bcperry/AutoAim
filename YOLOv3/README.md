# YOLOv3 in PyTorch

### Download pretrained weights on Pascal-VOC
Available on Kaggle: [link](https://www.kaggle.com/dataset/1cf520aba05e023f2f80099ef497a8f3668516c39e6f673531e3e47407c46694)

### Training
Edit the config.py file to match the setup you want to use. Then run train.py

## YOLOv3 paper
The implementation cloned from the [Machine Learning Collection by Aladdin Persson](https://github.com/aladdinpersson/Machine-Learning-Collection), based on the following paper:

### An Incremental Improvement 
by Joseph Redmon, Ali Farhadi

#### Abstract
We present some updates to YOLO! We made a bunch of little design changes to make it better. We also trained this new network that’s pretty swell. It’s a little bigger than last time but more accurate. It’s still fast though, don’t worry. At 320 × 320 YOLOv3 runs in 22 ms at 28.2 mAP, as accurate as SSD but three times faster. When we look at the old .5 IOU mAP detection metric YOLOv3 is quite good. It achieves 57.9 AP50 in 51 ms on a Titan X, compared to 57.5 AP50 in 198 ms by RetinaNet, similar performance but 3.8× faster. As always, all the code is online at https://pjreddie.com/yolo/.

```
@article{yolov3,
  title={YOLOv3: An Incremental Improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal = {arXiv},
  year={2018}
}
```
