# used by both the pipeline and the backend, so an image is resized the same way
# for training and for serving

import hashlib

import numpy as np
from PIL import Image


def to_model_input(img, size):
    # stays uint8, the model has its own Rescaling layer
    img = img.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def pixel_hash(img):
    # hash of the pixels, not the file: the feedback store re-saves uploads as PNG,
    # so a file hash would not match the original image anymore
    arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
    return hashlib.sha1(str(arr.shape).encode() + arr.tobytes()).hexdigest()
