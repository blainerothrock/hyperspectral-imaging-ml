# Tips and Tricks when Implementing HybridSN

## Conversion to PyTorch
### Why?
The main reason we converted this code was to learn. This paper's model seemed like a relatively simple model using 
out-of-the-box implementations of well studied components of deep learning. really just convolutional layers,
fully connected layers and dropout. Also, this model has a state of the art on a few of the Hyperspectral imaging
leaderboard on papers with code, something 100% accuracy. 

Our initial intent was to build a generalized model into a small framework that could easily be built upon with new
models and datasets. 

### Pytorch CrossEntropy Loss & Softmax
Not a issue with the original code, but something we ended up debugging for far too long. PyTorch's out of the box
[Cross Entropy Loss](https://pytorch.org/docs/stable/nn.html?highlight=crossentropyloss#torch.nn.CrossEntropyLoss) 
function calculates Softmax (log softmax). We were under the impression that we passed the probability into the 
function rather than computing on logits. This caused our loss to stall in training and followed some un-needed hours 
of debugging our input data. 

Looking at [Tensorflow's implementation](https://www.tensorflow.org/api_docs/python/tf/keras/losses/CategoricalCrossentropy)
it is calculated the same way, but does offer a `from_logits` boolean to turn off the internal softmax computation.

## Unit Tests
Testing code is always a good idea. Specifically, there are some complicated transformations done to the 
hyperspectral imaging and we wanted to generalize them for a framework. In order to do that, we needed to ensure
they were working as expected on toy data and unit tests help accomplish this. Most import was testing the transforms
required in pre-processing of the image data, which are PCA and the splitting method. Adding unit tests helpped us in two
ways. First, writing unit tests helps in understanding what the code is doing which can lead to enhacements. Second,
it really helps with debugging, if you can ensure a componenting is working independently of the training pipeline it can be ruled
out when debugging.  

### depth dimension & channels
PyTorch usually uses the depth dimensions as the first dimension in input, `(batch_size, channel, depth, width, height)`
, [see documentation for 3D convolutional layers](https://pytorch.org/docs/stable/nn.html?highlight=conv3d#torch.nn.Conv3d).
The Tensorflow implementation had depth as the last `(batch_size, height, width, depth)`. This caused some
confusion for us, although not related to the code simply the difference in the APIs. The main gotcha in this case is 
getting the convolution kernels correct based on the input configuration, where the dimensions need to match.

Although with this the channel dimensions caused some confusion because it's not really used in the model (1). The
confusion again came from the difference in APIs, Tensorflow usually places the channel dimension as the last
dimension in input ([see documentation](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv3D)) whereas
PyTorch uses the first dimension. When implementing this doesn't actually matter, any dimension can be used, but it 
can become confusing when reading documentation or examples from the respective frameworks.  

## Training
### Data Shuffling
Out initial training runs showed our training data overfitting very quickly and a increasing loss for the testing data.
We figured this was a bug and later realized that this was a result of not shuffling the image data before splitting
training and testing data. Randomly shuffling after generating the image cubes from the ordinal image allowed us to
see training results from the paper. We believe this problem was due to some distribution missing from the training
set given the ordering of the cubes after splitting. 

## Issues with original codebase
The original code base is implemented in a jupyter notebook [here](https://github.com/gokriznastic/HybridSN/blob/master/Hybrid-Spectral-Net.ipynb) using Keras. 
The code was in decent shape, but we decided to convert this to run in a framework like experiment in PyTorch 
(discussed above). We experienced the following when running the code in its given state:
* Conversion to Tensorflow 2
    - pretty much handles with import Keras from TensorFlow
* Many magic numbers
* Testing data mixed up with training data?
    - looking at cell 12, it seems the training and testing data are swapped with using about 30% as training data and
      70% and testing data. We assumed this was a mistake.
* Minor issue with saving the model state
    - `acc` versus `accurary` in the model checkpoint definition
