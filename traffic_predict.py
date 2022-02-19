import cv2
import csv
import numpy as np
import os
import sys
import tensorflow as tf

IMG_OUTPUT_WIDTH = 180
IMG_OUTPUT_HEIGHT = 180
MOSAIC_LENGTH = 6
TEXT_POSITION_H = 5
TEXT_POSITION_V = -10


def load_descriptions(csv_file):
    
    signs = {}

    with open(csv_file) as f:
        reader = csv.reader(f)
        header_row = next(reader)
        for row in reader:
            signs[int(row[0])] = row[1]

    return(signs)


def prep_images(directory):
    """Read image into csv and resize to match training images."""
    
    images = []

    for image in os.listdir(directory):
        img = cv2.imread(os.path.join(directory, image))
        img = cv2.resize(img, (30, 30)) # Match training image size
        images.append(
            {'name': image, 'array': img}
        )

    return images


def add_predictions_to_images(image_list, predictions):
    """Adds prediction and confidence level to each image dictionary."""

    for index, array in enumerate(predictions):
        prediction = np.argmax(array)
        confidence = "{:.0%}".format(array[prediction])
        image_list[index]['prediction'] = prediction
        image_list[index]['confidence'] = confidence

    return image_list


def add_text_to_images(image_list, sign_conversion):
    """Use cv2 library to superimpose prediciton results on each image."""

    for image in image_list:
        image['complete'] = cv2.putText(
            cv2.resize(image['array'], (180, 180)),
            sign_conversion[image['prediction']].upper(),
            (TEXT_POSITION_H, IMG_OUTPUT_HEIGHT + TEXT_POSITION_V),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 180, 0),
            thickness=1
        )

    return image_list


def create_mosaic(image_list):
    """Create mosiac from rows of 6 images."""

    final_bank = []
    # Group images into rows of MOSAIC_LENGTH
    for i in range(0, len(image_list), MOSAIC_LENGTH):
        # Create horizontal row using hconcat function
        h_row = [image['complete'] for image in image_list[i:i+MOSAIC_LENGTH]]
        # Add blank squares when the row has less than MOSAIC_LENGTH images
        while len(h_row) < MOSAIC_LENGTH:
            # Use numpy array of zeros to create blank images
            h_row.append(np.zeros((IMG_OUTPUT_WIDTH, IMG_OUTPUT_HEIGHT, 3),
                dtype=np.uint8))
        row = cv2.hconcat(h_row)
        final_bank.append(row)

    return cv2.vconcat(final_bank)


def main():

    if len(sys.argv) < 3:
        sys.exit("Usage: python traffic_predict.py model image_directory")

    # Load sign dictionary
    signs = load_descriptions('sign_descriptions.csv')

    # Load images
    images = prep_images(sys.argv[2])
    
    # Create list of image arrays for use with model.predict()
    image_arrays = [image['array'] for image in images]

    # Load existing tf model from file
    model = tf.keras.models.load_model(sys.argv[1])

    # Get predictions from model
    predictions = model.predict(np.array(image_arrays))

    # Add most-likely prediction and confidence level to each image dictionary
    images = add_predictions_to_images(images, predictions)

    # Superimpose text on images
    images = add_text_to_images(images, signs)

    # Create mosaic
    mosaic = create_mosaic(images)

    cv2.imshow('Images', mosaic)
    k = cv2.waitKey(0)

if __name__ == "__main__":
    main()