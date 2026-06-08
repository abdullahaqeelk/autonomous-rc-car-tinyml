import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

import params


IMG_W, IMG_H = params.inputres
CROP_Y1 = params.crop_y1
CROP_Y2 = params.crop_y2
TARGET_COLUMN = params.target_column
IMAGES_DIRNAME = params.images_dirname
LABELS_FILENAME = params.labels_filename
RUN_PREFIX = params.run_prefix

MODEL_BASENAME = "model_recommended_final_umar"
ZERO_CONTROL_FRAME_DROP_PERCENT = 100.0
ZERO_CONTROL_EPSILON = 1e-6

m = __import__(params.modelname)
model = m.model


def preprocess_frame(img):
    img = img[CROP_Y1:CROP_Y2, :]
    return cv2.resize(img, (IMG_W, IMG_H))


def iter_run_dirs(dataset_root):
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_root}")
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and path.name.startswith(RUN_PREFIX)
    )


def drop_zero_control_frames(df, drop_percent):
    if drop_percent <= 0:
        return df, 0

    required_columns = {"steering", "throttle"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns {sorted(missing)} needed for zero-control filtering"
        )

    steer = pd.to_numeric(df["steering"], errors="coerce").fillna(0.0)
    throttle = pd.to_numeric(df["throttle"], errors="coerce").fillna(0.0)
    zero_mask = (steer.abs() <= ZERO_CONTROL_EPSILON) & (
        throttle.abs() <= ZERO_CONTROL_EPSILON
    )

    zero_indices = df.index[zero_mask]
    zero_count = len(zero_indices)
    if zero_count == 0:
        return df, 0

    drop_fraction = min(max(drop_percent / 100.0, 0.0), 1.0)
    drop_count = int(round(zero_count * drop_fraction))
    if drop_count == 0:
        return df, 0

    drop_indices = np.random.choice(
        zero_indices.to_numpy(), size=drop_count, replace=False
    )
    filtered_df = df.drop(index=drop_indices)
    return filtered_df, drop_count


def load_labels(labels_file, zero_control_drop_percent=0.0):
    df = pd.read_csv(labels_file)
    df.columns = [c.lower().strip() for c in df.columns]

    required_columns = {"frame", TARGET_COLUMN}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)} in {labels_file}")

    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    df = df.dropna(subset=[TARGET_COLUMN])
    df, dropped_rows = drop_zero_control_frames(df, zero_control_drop_percent)
    if dropped_rows:
        print(
            f"Skipped {dropped_rows} zero-control frames from {Path(labels_file).parent.name}"
        )

    return {
        str(row["frame"]): float(np.clip(row[TARGET_COLUMN], -1.0, 1.0))
        for _, row in df.iterrows()
    }


def apply_random_blur(img):
    if np.random.rand() < 0.25:
        kernel_size = int(np.random.choice([3, 5]))
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    return img


def apply_random_brightness(img):
    alpha = float(np.random.uniform(0.75, 1.25))
    return cv2.convertScaleAbs(img, alpha=alpha, beta=0)


def augment_training_sample(img, label):
    samples = [(img, label)]

    if abs(label) > 0.03:
        flipped = cv2.flip(img, 1)
        samples.append((flipped, -label))

    bright = apply_random_brightness(img)
    samples.append((bright, label))

    blurred = apply_random_blur(img.copy())
    samples.append((blurred, label))

    return samples


def load_training_data(dataset_root):
    imgs, labels = [], []

    run_dirs = iter_run_dirs(dataset_root)
    if not run_dirs:
        raise RuntimeError(f"No run directories found under {dataset_root}")

    for run_dir in run_dirs:
        images_dir = run_dir / IMAGES_DIRNAME
        labels_file = run_dir / LABELS_FILENAME
        if not images_dir.exists() or not labels_file.exists():
            continue

        labels_by_frame = load_labels(
            labels_file,
            zero_control_drop_percent=ZERO_CONTROL_FRAME_DROP_PERCENT,
        )
        frame_files = sorted(
            path
            for path in images_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".jpg"
        )

        run_count = 0
        for frame_path in frame_files:
            frame_name = frame_path.name
            if frame_name not in labels_by_frame:
                continue

            img = cv2.imread(str(frame_path))
            if img is None:
                continue

            img = preprocess_frame(img)
            label = labels_by_frame[frame_name]

            for aug_img, aug_label in augment_training_sample(img, label):
                imgs.append(aug_img.astype(np.uint8))
                labels.append(np.float32(aug_label))
                run_count += 1

        print(f"Loaded {run_dir.name}: {run_count} training samples")

    X = np.array(imgs, dtype=np.uint8)
    y = np.array(labels, dtype=np.float32)

    if len(X) == 0:
        raise RuntimeError("No training samples found.")

    return X, y


def load_validation_data(run_dir):
    run_path = Path(run_dir)
    images_dir = run_path / IMAGES_DIRNAME
    labels_file = run_path / LABELS_FILENAME

    if not images_dir.exists() or not labels_file.exists():
        raise FileNotFoundError(
            f"Validation run is missing '{IMAGES_DIRNAME}' or '{LABELS_FILENAME}': {run_dir}"
        )

    labels_by_frame = load_labels(labels_file)

    imgs, labels = [], []
    for frame_path in sorted(images_dir.glob("*.jpg")):
        frame_name = frame_path.name
        if frame_name not in labels_by_frame:
            continue

        img = cv2.imread(str(frame_path))
        if img is None:
            continue

        imgs.append(preprocess_frame(img).astype(np.uint8))
        labels.append(np.float32(labels_by_frame[frame_name]))

    X = np.array(imgs, dtype=np.uint8)
    y = np.array(labels, dtype=np.float32)

    if len(X) == 0:
        raise RuntimeError(f"No validation samples found in {run_dir}")

    return X, y


def weighted_huber(y_true, y_pred):
    base = tf.keras.losses.huber(y_true, y_pred, delta=0.1)
    weights = 1.0 + 5.0 * tf.abs(y_true)
    return tf.reduce_mean(weights * base)


def main():
    np.random.seed(42)
    tf.random.set_seed(42)

    X_train, y_train = load_training_data(params.dataset_dir)
    X_val, y_val = load_validation_data(params.validation_run_dir)

    print("Training shape:", X_train.shape, y_train.shape)
    print("Validation shape:", X_val.shape, y_val.shape)
    print(f"Training memory: ~{X_train.nbytes / (1024 ** 3):.2f} GB")

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-4),
        # loss=weighted_huber,
        loss=weighted_huber,
        metrics=[tf.keras.metrics.MeanSquaredError(name="mse")],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f"{MODEL_BASENAME}.h5", monitor="val_loss", save_best_only=True
        ),
    ]

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=params.num_epochs,
        batch_size=256,
        shuffle=True,
        callbacks=callbacks,
    )

    model.save(f"{MODEL_BASENAME}.h5")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(f"{MODEL_BASENAME}.tflite", "wb") as f:
        f.write(tflite_model)

    os.makedirs("models", exist_ok=True)
    with open(os.path.join("models", f"{MODEL_BASENAME}.tflite"), "wb") as f:
        f.write(tflite_model)

    print(f"Saved {MODEL_BASENAME}.h5 and {MODEL_BASENAME}.tflite")


if __name__ == "__main__":
    main()
