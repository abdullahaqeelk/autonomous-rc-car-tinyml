modelname = "pilotnet"

# Match your training preprocessing
inputres = (68, 68)  # (width, height)
inputchannels = 3

# Shared preprocessing / label settings for training and validation
# crop_y1 = 55
# crop_y2 = 135
crop_y1 = 60
crop_y2 = 140
target_column = (
    "smooth_steer"  # Change to "smooth_steer" if you want actuator-smoothed labels
)

# Dataset layout
dataset_dir = "dataset"  # Root directory containing run subdirectories
images_dirname = "images"
labels_filename = "labels.csv"
run_prefix = "run_"
validation_run_dir = "run_006_val"  # <-- change this to the run you want to validate on
# run_004 run_006
# Training config
num_epochs = 25
