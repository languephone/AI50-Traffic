## Traffic

For this project, I started by using the same layers as the handwriting.py file from the lecture, which is to say:
* 1 2D convolutional layer of 32 filters
* 1 Max-Pooling layer with a pool size of 2x2
* 1 flattening layer
* 1 dense hidden layer of 128 neurons
* 1 dropout later with a rate of 50%

The results from this initial set of layers were abysmal: 5.3% accuracy.  I started playing around first with the dropout, bringing it down to 30%, and this had a dramatic effect on the model, immediately bringing the accuracy up to 92%.  Further changes to dropout had incrementally positive effects:
* 50% dropout: 5.3% accuracy
* 30% dropout: 91.7% accuracy
* 20% dropout: 92.3% accuracy
* 10% dropout: 94.3% accuracy
* 5% dropout: 94.6% accuracy

At 5% dropout, the model seemed to be over-fitting on the data, where the accuracy in the 10th epoch of fitting was reported as 97%, but the accuracy in the evaluation stage was only 94.6%.

After resetting back to 20% dropout, I tried increasing the number of neurons from 128 to 256, which nearly doubled the time it took to run the tests, but only improved accuracy from 92.2% to 95.0%.  Two layers of 128 neurons somehow was worse than a single layer, producing accuracy of 82.5%.

The next area of experimentation was with the convolutional & pooling layers.  Going back to 128 neurons, I added a second convolutional layer with 32 filters, and a second max-pooling layer of 2x2 pixels, which increased the accuracy to 97.6%.  Additional tweaks didn't really improve on that:
* 2x32 filters, 2x max-pooling of 2x2: 97.6% accuracy
* 2x32 filters, 1 max-pooling of 4x4, 1 max-pooling of 2x2: 91.6% accuracy
* 2x64 filters, 2x max-pooling of 2x2: 97.9% accuracy

Moving from 32 to 64 filters again nearly doubled the time to do the fitting, for only a 0.3% increase in accuracy.

Finally, I tried to combine the best aspects of previous test runs by using 256 neurons, 2x convolutional layers with 64 filters, 2x max-pooling layers, and a dropout rate of 20%.  This resulted in essentially the same accuracy as using the layers with 32 filters and only 128 neurons, but the longest time to fit at around 27 seconds per epoch.  For the submitted model, I went back to the version which used 32 filters and only 128 neurons because it offered essentially the best performance with the lowest time to fit.

-------------------------------------------------------------------------------

Despite completing the assignment with the model described above, it left me feeling distinctly unsatisfied.  I saw the numbers on the screen describe the accuracy, but the model still felt purely academic rather than a real prediction tool.  So, I decided I would use the saved model to try to predict the sign category of 'real' images.

To test the saved model on 'real' images, I created a new file named traffic_predict.py which loads the saved model and uses the model.predict method to generate the highest-probability category for any input image.  To get 'real' images, I used Google Maps to virually walk around Munich and take screen grabs of any traffic signs that I could find.  I saved these images into a folder which was used as input to the model.predict method.

I was also able to find a csv file online which translated the sign number categories into a description of the signs, and I used the cv2 library to generate a new image including the original image and text of the predicted category written on top.

Finally, I used the cv2 library to create a mosaic image of all of the images fed into the model.predict method along with the model's prediction.  This was a much more satisfying result, where I could see a 'real' image, not part of the original data set, and how the model was able to interpret the image.

![mosaic of signs](mosaic.png 'traffic signs')