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
        images.append(img)

    return images


def main():

    if len(sys.argv) < 2:
        sys.exit("Usage: python traffic_predict.py image_directory")

    # Load sign dictionary
    signs = load_descriptions('sign_descriptions.csv')

    # Load images
    to_test = prep_images(sys.argv[1])

    # Load existing tf model from file
    model = tf.keras.models.load_model("best_model.h5")

    # Get predictions from model
    predictions = model.predict(np.array(to_test))

    # Get most-likely prediction for each image
    for array in predictions:
        prediction = np.argmax(array)
        confidence = "{:.0%}".format(array[prediction])
        print(f'Prediction: {signs[prediction]} with confidence: {confidence}')


if __name__ == "__main__":
    main()