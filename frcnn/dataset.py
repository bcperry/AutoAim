import torch
import fiftyone.utils.coco as fouc
from PIL import Image
import random
import torchvision.transforms as transforms
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
import albumentations as A
import cv2

class FiftyOneTorchDataset(torch.utils.data.Dataset):
    """A class to construct a PyTorch dataset from a FiftyOne dataset.

    Args:
        fiftyone_dataset: a FiftyOne dataset or view that will be used for training or testing
        transforms (None): a list of PyTorch transforms to apply to images and targets when loading
        gt_field ("ground_truth"): the name of the field in fiftyone_dataset that contains the
            desired labels to load
        classes (None): a list of class strings that are used to define the mapping between
            class names and indices. If None, it will use all classes present in the given fiftyone_dataset.
    """

    def __init__(
            self,
            fiftyone_dataset,
            transforms=None,
            gt_field="ground_truth",
            classes=None,
    ):

        self.samples = fiftyone_dataset
        self.transforms = transforms
        self.gt_field = gt_field

        self.img_paths = self.samples.values("filepath")

        self.classes = classes
        if not self.classes:
            # Get list of distinct labels that exist in the view
            self.classes = self.samples.distinct(
                "%s.detections.label" % gt_field
            )

        if self.classes[0] != "background":
            self.classes = ["background"] + self.classes

        self.labels_map_rev = {c: i for i, c in enumerate(self.classes)}

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        sample = self.samples[img_path]
        metadata = sample.metadata
        img = cv2.imread(img_path)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

        # divide by 255
        img = img_rgb / 255.0


        boxes = []
        labels = []
        area = []
        iscrowd = []
        detections = sample[self.gt_field].detections
        for det in detections:
            category_id = self.labels_map_rev[det.label]
            coco_obj = fouc.COCOObject.from_label(
                det, metadata, category_id=category_id,
            )
            x, y, w, h = coco_obj.bbox
            boxes.append([x, y, x + w, y + h])
            labels.append(coco_obj.category_id)
            area.append(coco_obj.area)

        if self.transforms is not None:
            img_trans = self.transforms(image=img, bboxes=boxes, labels=labels)
            img = img_trans['image']
            boxes = torch.tensor(img_trans['bboxes'])
            labels = torch.tensor(img_trans['labels'])

        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target["image_id"] = torch.as_tensor([idx])
        target["area"] = torch.as_tensor(area, dtype=torch.float32)
        target["iscrowd"] = torch.as_tensor([0 for _ in labels], dtype=torch.int64)

        if len(target['boxes']) == 0:
            test = 1



        return img, target

    def __len__(self):
        return len(self.img_paths)



def get_transforms(train):
    if train:
        transform = A.Compose([A.HorizontalFlip(0.5), ToTensorV2(p=1.0) ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
    else:
        transform = A.Compose([ToTensorV2(p=1.0)])
    return transform

def collate_fn(batch):
    valid = []
    replace = []
    for image in range(len(batch)):
        if len(batch[image][1]['boxes']) != 0: #make sure the image has some bbox
            valid.append(image)
        else:
            replace.append(image)
    if len(valid) != len(batch):
        for bad in replace:
            batch[bad] = batch[random.choice(valid)]
    return tuple(zip(*batch))