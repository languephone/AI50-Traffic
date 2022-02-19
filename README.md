## Traffic

For this project, I started by using the same layers as the handwriting.py file from the lecture, which is to say:
* 1 2D convolutional layer of 32 filters
* 1 Max-Pooling layer with a pool size of 2x2
* 1 flattening layer
* 1 dense hidden layer of 128 neurons
* 1 dropout later with a rate of 50%

The results from this initial set of layers were atrocious: accuracy of 5.3%.

