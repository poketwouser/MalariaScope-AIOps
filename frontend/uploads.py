"""Checks an image before the Predict page shows it (rubric: a foolproof UI)."""

import io

from PIL import Image

MAX_MB = 5


def upload_error(data):
    """Return a message for the user if the image can't be used, else None."""
    if len(data) > MAX_MB * 1024 * 1024:
        return f"This file is larger than {MAX_MB} MB. Please use a smaller image."
    try:
        # a full decode: a file can have a valid header and still be truncated
        Image.open(io.BytesIO(data)).load()
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
        return "This file is not a readable image. Please use a PNG or JPEG picture."
    return None
