"""The MalariaScope CNN.

Input is a uint8 (size, size, 3) cell image. Scaling to [0, 1] and training-time
augmentation are layers inside the model, so serving sends raw pixels and gets
the same preprocessing as training (Rescaling is part of the saved model).
"""

import keras
from keras import layers

CLASS_NAMES = ("parasitized", "uninfected")  # TFDS label order: 0, 1


def build_model(image_size, filters, dense_units, dropout, augment, learning_rate):
    """Build and compile a small conv net with a 2-way softmax output."""
    inputs = keras.Input(shape=(image_size, image_size, 3), name="image")
    x = layers.Rescaling(1.0 / 255)(inputs)
    if augment:
        # cells have no fixed orientation, so flips and rotations are label-preserving
        x = layers.RandomFlip("horizontal_and_vertical")(x)
        x = layers.RandomRotation(0.25)(x)

    for n in filters:
        x = layers.Conv2D(n, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(len(CLASS_NAMES), activation="softmax", name="probs")(x)

    model = keras.Model(inputs, outputs, name="malariascope_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
