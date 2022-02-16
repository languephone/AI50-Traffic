import cv2
import csv
import numpy as np
import os
import sys
import tensorflow as tf



def load_descriptions(csv_file):
    
    signs = {}

    with open(csv_file) as f:
        reader = csv.reader(f)
        header_row = next(reader)
        for row in reader:
            signs[int(row[0])] = row[1]

    return(signs)


def prep_images(directory):
    
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
            cv2.resize(image['array'], (150, 150)),
            sign_conversion[image['prediction']].upper(),
            (0, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 180, 0),
            thickness=1
        )

    image_bank = np.concatenate(
        [image['complete'] for image in image_list], axis=1)
    
    cv2.imshow('Images', image_bank)
    k = cv2.waitKey(0)


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

    # Show prediction results for each image
    add_text_to_images(images, signs)

if __name__ == "__main__":
    main()