"""Class labels, kept free of heavy imports so the backend can use them."""

CLASS_NAMES = ("parasitized", "uninfected")  # TFDS label order: 0, 1
POSITIVE = CLASS_NAMES.index("parasitized")
