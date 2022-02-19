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