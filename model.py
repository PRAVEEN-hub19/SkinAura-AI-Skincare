"""Transfer-learning CNN used to classify skin-type images."""

from tensorflow.keras import applications, layers, models


def build_cnn(input_shape=(160, 160, 3), num_classes=3):
    """Build a compact CNN using ImageNet features and a skin-type head."""
    backbone = applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    backbone.trainable = False

    inputs = layers.Input(shape=input_shape, name="image")
    normalized = layers.Rescaling(1.0 / 127.5, offset=-1)(inputs)
    features = backbone(normalized, training=False)
    features = layers.GlobalAveragePooling2D()(features)
    features = layers.Dropout(0.35)(features)
    outputs = layers.Dense(num_classes, activation="softmax")(features)
    return models.Model(inputs, outputs, name="SkinAura_MobileNetV2")


if __name__ == "__main__":
    # Quick sanity check: prints the model summary when this file is run
    # directly (python model.py) so you can confirm TensorFlow is working
    # before starting a full training run.
    m = build_cnn(input_shape=(160, 160, 3))
    m.summary()
