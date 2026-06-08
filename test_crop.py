import cv2
import numpy as np
import pandas as pd
import os
from pathlib import Path

import params

# Load all images and labels from dataset
dataset_dir = params.dataset_dir
images_dirname = params.images_dirname
labels_filename = params.labels_filename
target_column = params.target_column
img_w, img_h = params.inputres

# Collect all images and labels
all_images = []
all_labels = []

for run_dir in sorted(Path(dataset_dir).glob(f"{params.run_prefix}*")):
    if not run_dir.is_dir():
        continue

    images_path = run_dir / images_dirname
    labels_path = run_dir / labels_filename

    if not images_path.exists() or not labels_path.exists():
        continue

    # Load labels
    df = pd.read_csv(labels_path)
    df.columns = [c.lower().strip() for c in df.columns]
    df[target_column] = pd.to_numeric(df[target_column], errors="coerce")
    labels_dict = {str(row["frame"]): row[target_column] for _, row in df.iterrows()}

    # Load images
    for img_file in sorted(images_path.glob("*.jpg")):
        frame_name = img_file.name
        if frame_name in labels_dict:
            all_images.append(str(img_file))
            all_labels.append(labels_dict[frame_name])

if not all_images:
    print("No images found in dataset!")
    exit(1)

print(f"Loaded {len(all_images)} images with labels")


def nothing(x):
    pass


# Create window with trackbars
cv2.namedWindow("Image Viewer")
cv2.createTrackbar("Image Index", "Image Viewer", 0, len(all_images) - 1, nothing)
cv2.createTrackbar("Top Crop", "Image Viewer", params.crop_y1, 400, nothing)
cv2.createTrackbar("Bottom Crop", "Image Viewer", params.crop_y2, 480, nothing)

print("Controls:")
print("  - Use 'Image Index' to browse images")
print("  - Adjust 'Top Crop' and 'Bottom Crop' to tune crop region")
print("  - Press 'q' to exit")

while True:
    # Get trackbar positions
    idx = cv2.getTrackbarPos("Image Index", "Image Viewer")
    top = cv2.getTrackbarPos("Top Crop", "Image Viewer")
    bottom = cv2.getTrackbarPos("Bottom Crop", "Image Viewer")

    # Ensure bottom > top
    if bottom <= top:
        bottom = top + 1

    # Load image
    frame = cv2.imread(all_images[idx])
    if frame is None:
        continue

    label = all_labels[idx]

    # Show crop bounds on original
    preview_img = frame.copy()
    cv2.line(preview_img, (0, top), (frame.shape[1], top), (0, 255, 0), 2)
    cv2.line(preview_img, (0, bottom), (frame.shape[1], bottom), (0, 0, 255), 2)
    cv2.putText(
        preview_img,
        f"Steering: {label:.3f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        preview_img,
        "MODEL SEEING AREA",
        (10, top - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
    )

    # Perform crop and resize
    cropped = frame[top:bottom, :]
    model_input = cv2.resize(cropped, (img_w, img_h))

    # Show windows
    cv2.imshow("Image Viewer", preview_img)
    cv2.imshow(
        "Model Input", cv2.resize(model_input, (400, 132))
    )  # Upscaled for visibility

    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()

print(f"\nFinal crop values:")
print(f"CROP_Y1 = {top}")
print(f"CROP_Y2 = {bottom}")
