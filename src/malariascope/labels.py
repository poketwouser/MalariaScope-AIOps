# no heavy imports here, the backend imports this too
CLASS_NAMES = ("parasitized", "uninfected")  # TFDS label order: 0, 1
POSITIVE = CLASS_NAMES.index("parasitized")
