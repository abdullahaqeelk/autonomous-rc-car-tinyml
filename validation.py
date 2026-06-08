import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ai_edge_litert.interpreter import Interpreter
import params


MODEL_PATH = "model_recommended_final_umar.tflite"
RUN_DIR = params.validation_run_dir
BASE_ALPHA = 0.12
TURN_ALPHA_GAIN = 0.38
TURN_THRESHOLD = 0.05

IMG_W, IMG_H = params.inputres
CROP_Y1 = params.crop_y1
CROP_Y2 = params.crop_y2
TARGET_COLUMN = params.target_column
IMAGES_DIRNAME = params.images_dirname
LABELS_FILENAME = params.labels_filename


def preprocess_frame(img):
    img = img[CROP_Y1:CROP_Y2, :]
    return cv2.resize(img, (IMG_W, IMG_H))


def load_labels(labels_path):
    df = pd.read_csv(labels_path)
    df.columns = [c.lower().strip() for c in df.columns]

    required_columns = {"frame", TARGET_COLUMN}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)} in {labels_path}")

    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    df = df.dropna(subset=[TARGET_COLUMN])

    return {
        str(row["frame"]): float(np.clip(row[TARGET_COLUMN], -1.0, 1.0))
        for _, row in df.iterrows()
    }

past=0
def main():
    images_dir = os.path.join(RUN_DIR, IMAGES_DIRNAME)
    labels_path = os.path.join(RUN_DIR, LABELS_FILENAME)

    if not os.path.isdir(images_dir) or not os.path.exists(labels_path):
        raise FileNotFoundError(f"Validation data missing in '{RUN_DIR}'.")

    labels_by_frame = load_labels(labels_path)
    frame_files = sorted(
        os.path.join(images_dir, name)
        for name in os.listdir(images_dir)
        if name.lower().endswith(".jpg")
    )

    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    inp_idx = input_details["index"]
    out_idx = output_details["index"]
    inp_dtype = input_details["dtype"]

    raw_predictions = []
    smooth_predictions = []
    actuals = []
    last_smoothed = 0.0

    print(f"Evaluating run: {RUN_DIR}")
    print(f"Model: {MODEL_PATH}")
    print(
        f"Adaptive smoothing: alpha = {BASE_ALPHA:.2f} + "
        f"{TURN_ALPHA_GAIN:.2f} * abs(raw_pred)"
    )

    for frame_path in frame_files:
        frame_name = os.path.basename(frame_path)
        if frame_name not in labels_by_frame:
            continue

        img = cv2.imread(frame_path)
        if img is None:
            continue

        img = preprocess_frame(img).astype(inp_dtype)
        img = np.expand_dims(img, axis=0)

        interpreter.set_tensor(inp_idx, img)
        interpreter.invoke()
        raw_pred = float(interpreter.get_tensor(out_idx)[0][0])

        if not smooth_predictions:
            smoothed = raw_pred
        else:
            turn_strength = min(abs(raw_pred), 1.0)
            alpha = BASE_ALPHA + (TURN_ALPHA_GAIN * turn_strength)
            smoothed = alpha * raw_pred + (1 - alpha) * last_smoothed

        last_smoothed = smoothed
        raw_predictions.append(raw_pred)
        smooth_predictions.append(smoothed)
        actuals.append(labels_by_frame[frame_name])

    if not raw_predictions:
        raise RuntimeError(f"No validation samples found in '{RUN_DIR}'.")

    raw_predictions = np.array(raw_predictions, dtype=np.float32)
    smooth_predictions = np.array(smooth_predictions, dtype=np.float32)
    actuals = np.array(actuals, dtype=np.float32)

    raw_mse = np.mean((raw_predictions - actuals) ** 2)
    smooth_mse = np.mean((smooth_predictions - actuals) ** 2)
    raw_corr = np.corrcoef(actuals, raw_predictions)[0, 1]
    smooth_corr = np.corrcoef(actuals, smooth_predictions)[0, 1]

    left_mask = actuals < -TURN_THRESHOLD
    right_mask = actuals > TURN_THRESHOLD
    straight_mask = ~left_mask & ~right_mask

    def masked_mse(preds, truth, mask):
        if not np.any(mask):
            return float("nan")
        return float(np.mean((preds[mask] - truth[mask]) ** 2))

    print("\nPerformance")
    print(f"Raw MSE:       {raw_mse:.5f}")
    print(f"Smoothed MSE:  {smooth_mse:.5f}")
    print(f"Raw Corr:      {raw_corr:.3f}")
    print(f"Smoothed Corr: {smooth_corr:.3f}")
    print("\nBy turn direction")
    print(f"Left-turn MSE:     {masked_mse(raw_predictions, actuals, left_mask):.5f}")
    print(f"Right-turn MSE:    {masked_mse(raw_predictions, actuals, right_mask):.5f}")
    print(f"Straight-run MSE:  {masked_mse(raw_predictions, actuals, straight_mask):.5f}")

    plt.figure(figsize=(14, 7))
    plt.plot(actuals, label="Ground truth", color="#2b6cb0", linewidth=2, alpha=0.7)
    plt.plot(raw_predictions, label="Raw prediction", color="#d97706", linewidth=1.5, alpha=0.6)
    plt.plot(
        smooth_predictions,
        label="Smoothed prediction (adaptive alpha)",
        color="#c53030",
        linewidth=2,
    )
    plt.axhline(0, color="black", linestyle="--", alpha=0.3)
    plt.title(f"Steering validation: {os.path.basename(RUN_DIR)}")
    plt.xlabel("Frame")
    plt.ylabel("Steering")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="upper right")
    plt.show()


if __name__ == "__main__":
    main()
