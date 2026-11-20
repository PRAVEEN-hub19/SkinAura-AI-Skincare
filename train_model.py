"""
train_model.py
---------------
Trains the SkinAura CNN on the Oily/Dry/Normal Skin Types dataset and
saves the trained model + class label mapping to saved_model/.

HOW TO RUN (Windows, Anaconda Prompt):
    conda activate skinaura        (or your environment name)
    cd path\\to\\SkinAura
    python train_model.py

Make sure you have already:
    1. Downloaded the dataset from Kaggle (see config.py for the link).
    2. Placed it under SkinAura/dataset/ following the folder layout
       described in config.py.
    3. Installed dependencies:  pip install -r requirements.txt
"""

import os
import json

import config


def get_training_source_dir():
    """Prefer the prepared split; otherwise accept a complete raw train split."""
    expected_classes = {"dry", "normal", "oily"}
    for candidate in (config.PREPARED_TRAIN_DIR, config.TRAIN_DIR):
        if not os.path.isdir(candidate):
            continue
        train_classes = {
            name.lower()
            for name in os.listdir(candidate)
            if os.path.isdir(os.path.join(candidate, name))
        }
        if expected_classes.issubset(train_classes):
            return candidate
    return None


def get_data_generators():
    """
    Builds Keras ImageDataGenerators for training and validation.

    Prepared data uses its own validation split. Raw complete training data
    uses a reserved validation subset, never the held-out test split.
    """
    source_dir = get_training_source_dir()
    if source_dir is None:
        raise FileNotFoundError(
            "No directory contains all required classes: dry, normal, oily."
        )
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    if source_dir == config.PREPARED_TRAIN_DIR:
        train_datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            brightness_range=(0.75, 1.25),
            channel_shift_range=12.0,
        )
        val_datagen = ImageDataGenerator()

        train_gen = train_datagen.flow_from_directory(
            source_dir,
            target_size=config.IMG_SIZE,
            batch_size=config.BATCH_SIZE,
            class_mode="categorical",
            classes=config.CLASS_NAMES,
            shuffle=True,
        )
        val_gen = val_datagen.flow_from_directory(
            config.PREPARED_VALIDATION_DIR,
            target_size=config.IMG_SIZE,
            batch_size=config.BATCH_SIZE,
            class_mode="categorical",
            shuffle=False,
        )
    else:
        print("[INFO] Using a validation subset of the complete raw train/ folder.")
        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            brightness_range=(0.75, 1.25),
            channel_shift_range=12.0,
            validation_split=config.VALIDATION_SPLIT,
        )

        train_gen = datagen.flow_from_directory(
            source_dir,
            target_size=config.IMG_SIZE,
            batch_size=config.BATCH_SIZE,
            class_mode="categorical",
            classes=config.CLASS_NAMES,
            subset="training",
            shuffle=True,
        )
        val_gen = datagen.flow_from_directory(
            source_dir,
            target_size=config.IMG_SIZE,
            batch_size=config.BATCH_SIZE,
            class_mode="categorical",
            classes=config.CLASS_NAMES,
            subset="validation",
            shuffle=False,
        )

    return train_gen, val_gen


def get_test_generator():
    """Load a held-out test directory, when a complete split is available."""
    source_dir = get_training_source_dir()
    test_dir = (
        config.PREPARED_TEST_DIR
        if source_dir == config.PREPARED_TRAIN_DIR
        else config.TEST_DIR
    )
    expected_classes = set(config.CLASS_NAMES)
    test_classes = {
        name.lower()
        for name in os.listdir(test_dir)
        if os.path.isdir(os.path.join(test_dir, name))
    } if os.path.isdir(test_dir) else set()
    if not expected_classes.issubset(test_classes):
        return None
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    return ImageDataGenerator().flow_from_directory(
        test_dir,
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode="categorical",
        classes=config.CLASS_NAMES,
        shuffle=False,
    )


def plot_history(history):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(history.history["accuracy"], label="train_acc")
    axes[0].plot(history.history["val_accuracy"], label="val_acc")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train_loss")
    axes[1].plot(history.history["val_loss"], label="val_loss")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(config.HISTORY_PLOT_PATH)
    print(f"[INFO] Training curves saved to {config.HISTORY_PLOT_PATH}")


def main():
    os.makedirs(config.MODEL_DIR, exist_ok=True)

    if get_training_source_dir() is None:
        raise FileNotFoundError(
            f"No complete dataset found under {config.DATASET_DIR}.\n"
            "Training needs dry, normal, and oily images in train/. For the "
            "included dataset, run: python prepare_dataset.py, then retry. "
            "Source images are split by original image ID to prevent leakage."
        )

    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.optimizers import Adam

    from model import build_cnn

    train_gen, val_gen = get_data_generators()
    test_gen = get_test_generator()

    with open(config.LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(train_gen.class_indices, f, indent=2)
    print(f"[INFO] Class labels: {train_gen.class_indices}")

    model = build_cnn(
        input_shape=config.IMG_SIZE + (3,),
        num_classes=len(train_gen.class_indices),
    )
    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
        ModelCheckpoint(config.MODEL_PATH, monitor="val_loss",
                         save_best_only=True, verbose=1),
    ]

    class_counts = {index: 0 for index in train_gen.class_indices.values()}
    for class_index in train_gen.classes:
        class_counts[class_index] += 1
    total_samples = sum(class_counts.values())
    class_weights = {
        class_index: total_samples / (len(class_counts) * count)
        for class_index, count in class_counts.items()
        if count
    }
    print(f"[INFO] Class weights: {class_weights}")

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
    )

    if test_gen is not None:
        test_loss, test_accuracy = model.evaluate(test_gen, verbose=0)
        print(f"[INFO] Held-out test accuracy: {test_accuracy:.3f}; loss: {test_loss:.3f}")

    print(f"[INFO] Best validation-loss model saved to {config.MODEL_PATH}")

    plot_history(history)


if __name__ == "__main__":
    main()
