"""
This file evaluates the yolo YOLOmodel on test data that was held out during YOLOmodel training.
"""

import torch
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Yolov1
from dataset import VOCDataset
from utils import (
    non_max_suppression,
    cellboxes_to_boxes,
    plot_image,
    load_checkpoint,
    plot_targets,
    pick_targets,
)

seed = 123
torch.manual_seed(seed)

# Hyperparameters etc.
LEARNING_RATE = 2e-9
DEVICE = "cuda" if torch.cuda.is_available else "cpu"
BATCH_SIZE = 5
WEIGHT_DECAY = 0
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL_FILE = "halo_model.pth.tar"
IMG_DIR = "halo_data"
LABEL_DIR = "halo_data/labels"


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes


transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])


def main():
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # load the pretrained YOLOmodel
    load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    test_dataset = VOCDataset(
        "train.csv",
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=False,
    )

    for x, y in test_loader:
        x = x.to(DEVICE)
        for idx in range(len(x)):
            bboxes = cellboxes_to_boxes(model(x))
            bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
            im = x[idx].permute(1,2,0).to("cpu")
            primary, secondary = pick_targets(im, bboxes)
            plot_targets(im, primary, secondary)
            #plot_image(x[idx].permute(1,2,0).to("cpu"), bboxes)


if __name__ == "__main__":
    main()