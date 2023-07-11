from PIL import Image, ImageDraw
import torchvision

# def render(image, predictions, labels, iou_threshold=.4):

#     nms = torchvision.ops.batched_nms(boxes=predictions['boxes'], scores=predictions['scores'], idxs=predictions['labels'], iou_threshold=iou_threshold) 
#     draw = ImageDraw.Draw(image)

#     idx = 0
#     for score, label, box, idx in zip(predictions["scores"], predictions["labels"], predictions["boxes"], range(len(predictions['scores']))):
#         if idx in nms:
#             box = [round(i, 2) for i in box.tolist()]
#             draw.rectangle(box, fill=None)
#             draw.text(box[:2], model.config.id2label[label.item()])
#             draw.text(box[2:], str(round(score.item(),2)))

#     return(image)

def render(image, predictions, labels):


    draw = ImageDraw.Draw(image)

    for bbox in predictions.cpu().numpy():
        box = [round(i, 2) for i in bbox[:4]]
        draw.rectangle(box, fill=None)
        draw.text(bbox[:2], labels[bbox[-1]])

    return(image)