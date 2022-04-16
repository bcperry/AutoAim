"""
This file is modified from Aladdin Persson's implementation of YOLO V1 for use on the Pascal VOC dataset.

ref: https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/object_detection/YOLO
"""

import torch
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import Yolov1
from dataset import VOCDataset
from utils import (
    gen_data_csv,
    mean_average_precision,
    get_bboxes,
    save_checkpoint,
    load_checkpoint,
)
from loss import YoloLoss

seed = 123
torch.manual_seed(seed)

# Hyperparameters etc.
LEARNING_RATE = 2e-9
DEVICE = "cuda" if torch.cuda.is_available else "cpu"
BATCH_SIZE = 5
WEIGHT_DECAY = 0
EPOCHS = 50
NUM_WORKERS = 2
PIN_MEMORY = True

# IF LOAD_MODEL IS SET TO FALSE IT WILL OVERWRITE PREVIOUS WEIGHTS
LOAD_MODEL = True # true continues training from previous wieghts file, False initializes training with random weights

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


def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss=loss.item())

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")


def main():
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = YoloLoss()

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    gen_data_csv(IMG_DIR, LABEL_DIR)

    train_dataset = VOCDataset(
        "train.csv",
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    # this is the starting MAP, any model that gets a better MAP than this will be saved over the current weights file
    best_map = 0.5
    for epoch in range(EPOCHS):
        print(f"Epoch: {epoch}")

        pred_boxes, target_boxes = get_bboxes(
            train_loader, model, iou_threshold=0.5, threshold=0.4
        )

        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )
        print(f"Train mAP: {mean_avg_prec}")

        if mean_avg_prec > best_map:
            best_map = mean_avg_prec
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)

        train_fn(train_loader, model, optimizer, loss_fn)


if __name__ == "__main__":
    main()