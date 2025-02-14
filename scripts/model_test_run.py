import tensorflow as tf
import cv2
import numpy as np
from object_detection.utils import label_map_util

# Load the label map
label_map_path = "./mtsd_data/mtsd_label_map.pbtxt"
category_index = label_map_util.create_category_index_from_labelmap(label_map_path, use_display_name=True)

# Load the saved model
model_name = "ssd_mobilenet_v2_320x320_coco17_tpu-8" # change the model here
detect_fn = tf.saved_model.load("./models/" + model_name + "/exported_model/saved_model")

def draw_boxes_with_labels(image, boxes, classes, scores, category_index, min_score_thresh=0.4):
    """
    Draws bounding boxes and labels inside the boxes on the given image.

    Args:
        image: The image to draw on.
        boxes: Bounding box coordinates (normalized) from the model.
        classes: Detected class indices.
        scores: Detection scores.
        category_index: Mapping of class indices to class names.
        min_score_thresh: Minimum score threshold for displaying detections.

    Returns:
        Annotated image with bounding boxes and labels inside.
    """
    height, width, _ = image.shape

    for i in range(len(boxes)):
        if scores[i] >= min_score_thresh:
            # Scale the box to image dimensions
            ymin, xmin, ymax, xmax = boxes[i]
            xmin = int(xmin * width)
            xmax = int(xmax * width)
            ymin = int(ymin * height)
            ymax = int(ymax * height)

            # Draw the rectangle (bounding box)
            color = (0, 255, 0)  # Green
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)

            # Get the label text
            class_id = int(classes[i])
            class_name = category_index[class_id]['name']
            label = f"{class_name}: {scores[i]:.2f}"

            # Calculate text size
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]

            # Define text background position
            text_x = xmin
            text_y = ymin - 5 if ymin > 20 else ymin + text_size[1] + 5
            text_bg_x = text_x + text_size[0] + 6
            text_bg_y = text_y - text_size[1] - 4

            # Draw text background (limegreen)
            cv2.rectangle(image, (text_x - 2, text_y - text_size[1] - 4), 
                          (text_bg_x, text_y + 2), (50, 205, 50), -1)

            # Draw label text (black)
            cv2.putText(
                image,
                label,
                (text_x, text_y),
                font,
                font_scale,
                (0, 0, 0), 
                font_thickness,
                lineType=cv2.LINE_AA,
            )

    return image

# Load an image
image_path = "./mtsd_data/mtsd_test_images/"
img = "P1840464_jpg.rf.8260bff9f8f1edfa5b332239dd9f9732.jpg" # change the image here
image = cv2.imread(image_path + img)

# Convert the image to a tensor
input_tensor = tf.convert_to_tensor(image)
input_tensor = input_tensor[tf.newaxis, ...]

# Perform detection
detections = detect_fn(input_tensor)

# Extract detection details
num_detections = int(detections.pop('num_detections'))
detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
boxes = detections['detection_boxes']
scores = detections['detection_scores']
classes = detections['detection_classes'].astype(np.int32)

# Draw detections on the image
annotated_image = draw_boxes_with_labels(
    image=image.copy(), 
    boxes=boxes, 
    classes=classes, 
    scores=scores, 
    category_index=category_index, 
    min_score_thresh=0.4
)

# Display the annotated image
cv2.imshow("Detections", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()