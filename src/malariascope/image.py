"""Image preprocessing shared by the pipeline and the inference backend.

Both sides call to_model_input, so a cell image is prepared the same way at
training and at serving time.
"""

import numpy as np
from PIL import Image


def to_model_input(img, size):
    """Convert a PIL image to a (size, size, 3) uint8 array.

    Scaling to [0, 1] happens inside the model (a Rescaling layer), so the array
    stays uint8 here and on the wire.
    """
    img = img.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.uint8)
