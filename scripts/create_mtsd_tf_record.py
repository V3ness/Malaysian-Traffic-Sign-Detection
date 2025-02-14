"""
commands:
python create_mtsd_tf_record.py --data_dir="E:\Git Repos\traffic-sign-detection\mtsd_data" --output_dir="E:\Git Repos\traffic-sign-detection\mtsd_data"
"""

import hashlib
import io
import logging
import os

import PIL.Image
import tensorflow as tf

import sys
sys.path.append('..')

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

import pandas as pd

flags = tf.compat.v1.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw dataset.')
flags.DEFINE_string('output_dir', '', 'Path to directory to output TFRecords.')
flags.DEFINE_string('label_map_path', r'E:\Git Repos\traffic-sign-detection\mtsd_data\mtsd_label_map.pbtxt',
                    'Path to label map proto')
FLAGS = flags.FLAGS

def df_to_tf_example(data, label_map_dict, image_subdirectory):
    """Convert CSV-derived dict to tf.Example proto.

    Args:
        data: dict holding CSV fields for a single image.
        label_map_dict: A map from string label names to integers ids.
        image_subdirectory: String specifying subdirectory within the dataset directory holding the images.

    Returns:
        example: The converted tf.Example.
    """
    img_path = os.path.join(image_subdirectory, data['filename'])
    with tf.io.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    width, height = image.size

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    for obj in data['object']:
        xmin.append(float(obj['xmin']) / width)
        ymin.append(float(obj['ymin']) / height)
        xmax.append(float(obj['xmax']) / width)
        ymax.append(float(obj['ymax']) / height)
        class_name = obj['class']
        classes_text.append(class_name.encode('utf8'))
        classes.append(label_map_dict[class_name])

    example = tf.train.Example(features=tf.train.Features(
        feature={
                'image/height': dataset_util.int64_feature(height),
                'image/width': dataset_util.int64_feature(width),
                'image/filename': dataset_util.bytes_feature(data['filename'].encode('utf8')),
                'image/source_id': dataset_util.bytes_feature(data['filename'].encode('utf8')),
                'image/encoded': dataset_util.bytes_feature(encoded_jpg),
                'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
                'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
                'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
                'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
                'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
                'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
                'image/object/class/label': dataset_util.int64_list_feature(classes),
        }))
    return example

def create_tf_record(output_filename, label_map_dict, csv_path, image_dir, examples):
    """Creates a TFRecord file from examples.

    Args:
        output_filename: Path to where output file is saved.
        label_map_dict: The label map dictionary.
        csv_path: Path to the CSV file containing annotations.
        image_dir: Directory where image files are stored.
        examples: List of image filenames to include in the TFRecord.
    """
    writer = tf.io.TFRecordWriter(output_filename)

    # Read the CSV file
    df = pd.read_csv(csv_path)

    for idx, example in enumerate(examples):
        if idx % 100 == 0:
            logging.info('On image %d of %d', idx, len(examples))

        # Filter rows for the current image
        image_data = df[df['filename'] == example]
        if image_data.empty:
            logging.warning(f"No annotations found for image: {example}")
            continue

        data = {
            'filename': example,
            'object': []
        }

        # Add bounding box and class information
        for _, row in image_data.iterrows():
            data['object'].append({
                'xmin': row['xmin'],
                'ymin': row['ymin'],
                'xmax': row['xmax'],
                'ymax': row['ymax'],
                'class': row['class']
            })

        # Convert to TFExample and write to TFRecord
        tf_example = df_to_tf_example(data, label_map_dict, image_dir)
        writer.write(tf_example.SerializeToString())

    writer.close()

def main(_):
    data_dir = FLAGS.data_dir
    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

    logging.info('Reading from dataset.')
    image_dir = os.path.join(data_dir, 'mtsd_train_images')
    csv_path = os.path.join(data_dir, 'train_dataset.csv')

    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_path)

    # List all unique image filenames
    examples_list = df['filename'].unique().tolist()

    # Split into train and validation sets (1600 train, 400 val) 80% Training, 20% Validation
    num_train = 1600
    train_examples = examples_list[:num_train]
    val_examples = examples_list[num_train:]

    logging.info('%d training and %d validation examples.', len(train_examples), len(val_examples))

    # Output paths
    train_output_path = os.path.join(FLAGS.output_dir, 'mtsd_train.record')
    val_output_path = os.path.join(FLAGS.output_dir, 'mtsd_val.record')

    # Create TFRecords
    create_tf_record(train_output_path, label_map_dict, csv_path, image_dir, train_examples)
    create_tf_record(val_output_path, label_map_dict, csv_path, image_dir, val_examples)

if __name__ == '__main__':
    tf.compat.v1.app.run()