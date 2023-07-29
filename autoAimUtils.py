'''various utilities for use with the AutoAim script'''
import ctypes
import PIL
from PIL import Image

def render(image: Image, predictions: list, labels: list) -> Image:
    """gets an image, and predictions and returns an updated image

    Parameters
    ----------
    iamge : Pillow Image instance
        The file location of the spreadsheet
    predictions : bool, optional
        A flag used to print the columns to the console (default is
        False)

    Returns
    -------
    list
        an image with the bboxes and predictions visible
    """

    draw = PIL.ImageDraw.Draw(image)

    for bbox in predictions.cpu().numpy():
        box = [round(i, 2) for i in bbox[:4]]
        draw.rectangle(box, fill=None)
        draw.text(bbox[:2], labels[bbox[-1]])

    return image

def getScreenInfo(scale: float = 1) -> dict['top': int, 'left': int, 'width': int, 'height': int]:
    """defines the size of the screen given a scaling factor

    Parameters
    ----------
    scale : float
        What percentage of the screen the user wishes to see

    Returns
    -------
    list
        a dict containging the top left pixel, along with a width and height
    """
    W, H = ctypes.windll.user32.GetSystemMetrics(0), ctypes.windll.user32.GetSystemMetrics(1)

    # find the center point of the screen
    x_center = W/2
    y_center = H/2

    # we need to define the top, bottom, left, and right bounds of the box in pixels
    top = int(y_center - (H * scale)/2)
    left = int(x_center - (W * scale)/2)
    bottom = int(y_center + (H * scale)/2)
    right = int(x_center + (W * scale)/2)

    width = int(right - left)
    height = int(bottom - top)

    # we save these as a dictionary to be used with
    monitor = {'top': top, 'left': left, 'width': width, 'height': height}

    return monitor
