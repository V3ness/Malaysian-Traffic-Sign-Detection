import os
import tensorflow as tf
import pandas as pd
import numpy as np
from PIL import Image
import pickle

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Path to test images and CSV dataset
DATASET_PATH = './mtsd_data/mtsd_test_images/'
CSV_PATH = './mtsd_data/test_dataset.csv'

# Load ground truht CSV file
gt_data = pd.read_csv(CSV_PATH)

# Convert CSV filename column to dictionary format
ground_truths = {}

for _, row in gt_data.iterrows():
    filename = row['filename']
    bbox = [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
    class_name = row['class']

    if filename not in ground_truths:
        ground_truths[filename] = {'gt_bboxes': [], 'gt_classes': []}

    ground_truths[filename]['gt_bboxes'].append(bbox)
    ground_truths[filename]['gt_classes'].append(class_name)

MODEL_NAME = 'ssd_mobilenet_v2_320x320_coco17_tpu-8\exported_model' # change the model here

# Define model path
MODEL_PATH = os.path.join('models', MODEL_NAME)
model = tf.saved_model.load(os.path.join(MODEL_PATH, 'saved_model'))

# Get all image filenames
test_images = [os.path.join(DATASET_PATH, f) for f in os.listdir(DATASET_PATH) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

# Dictionary to store results
detections = {}

def load_image_into_numpy_array(image_path):
    """ Load image and convert to numpy array """
    image = Image.open(image_path)
    image = image.convert('RGB')
    return np.array(image)

for image_path in test_images:
    image_np = load_image_into_numpy_array(image_path)

    # Convert image to tensor
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = tf.expand_dims(input_tensor, axis=0)

    # Run inference
    detections_result = model.signatures['serving_default'](input_tensor)

    # Extract filename without path
    filename = os.path.basename(image_path)

    # Store results with updated key names
    detections[filename] = {
        'pred_bboxes': detections_result['detection_boxes'].numpy().tolist(), 
        'pred_classes': detections_result['detection_classes'].numpy().astype(int).tolist(),
        'confidences': detections_result['detection_scores'].numpy().tolist(),
    }

    # Attach ground truth bounding boxes if available
    if filename in ground_truths:
        detections[filename]['gt_bboxes'] = ground_truths[filename]['gt_bboxes']
        detections[filename]['gt_classes'] = ground_truths[filename]['gt_classes']
    else:
        detections[filename]['gt_bboxes'] = []
        detections[filename]['gt_classes'] = []

print(f"Processed {len(detections)} test images")

# Save detections to pickle file
PICKLE_PATH = os.path.join(MODEL_PATH, 'detections_output_result.pkl')

with open(PICKLE_PATH, 'wb') as f:
    pickle.dump(detections, f)

print(f"Detections saved to {PICKLE_PATH}")