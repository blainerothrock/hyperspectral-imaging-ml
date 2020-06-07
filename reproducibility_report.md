**Overview**

Our goal was to explore a state of the art paper for classification of hyperspectral imaging. The paper [HybridSN: Exploring 3D-2D CNN Feature Hierarchy for Hyperspectral Image Classification](https://arxiv.org/pdf/1902.06701v3.pdf) is identified as the state-of-the-art in the [hyperspectral classification leaderboards on papers with code](https://paperswithcode.com/task/hyperspectral-image-classification). Our main interest in this paper was the fact that it claimed nearly 100% accuracy on 3 baseline datasets and that the model trains very quickly (in about 15 minutes in a consumer GPU). The model, HybridSN, uses both 3D and 2D convolutional layers. The appeal of this novel model is that it reduces the complexity of full 3D-convolutional models, but seemly still captures some of the feature extraction capabilities. Previous research has used either 2D or 3D convoluational layers.

We were able to reproduce the results of this paper on the Indian Pine dataset. We also added additional logging that further explores the quick convergence and hints that the model is overfitting and potentially memorizing the dataset. Moving forward we would like to explore simplifying the model and training on new datasets. The paper uses 3 datasets, but they are roughly the same size and contain similar classification and therefore perform similar.

**Dataset**

We developed our model using the [Indian Pine dataset](http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes) which is a single hyperspectral image (145, 145, 200) with reflective wavelengths in the range 0.4–2.5 10<sup>-6</sup> meters. This data was captured by NASA’s AVIRIS hyperspectral imaging aircraft flying over farmland in northwest Indiana. Each pixel is labeled as 1 of 16 classes depicting the type of surface with a additional unknown class.

*   Alfalfa
*   Corn-notill
*   Corn-mintill
*   Corn
*   Grass-pasture
*   Grass-trees
*   Grass-pasture-mowed
*   Hay-windrowed
*   Oats
*   Soybean-notill
*   Soybean-mintill
*   Soybean-clean
*   Wheat
*   Woods	
*   Buildings-Grass-Trees-Drives
*   Stone-Steel-Towers
*   Unknown (removed from the dataset)

The paper processes this into smaller hyperspectral imaging cubes, implementation discussed below, which results in roughly 10.5K images. Although these images are fairly large (25, 25, 30), the number of Samples is small given the model size of over 5MM parameters, hence our assitpion of overfitting, discussed more below. Given our research, there are not many labeled hyperspectral imaging datasets and Indian Pine seems to be the most popular for machine learning baselines.

**Dataset Implementation**

Reproduction of this paper seemed reasonable given the details provided in the paper, the computing resources required, and given [code base]() from the authors. We did not want to simply reproduce the results, but build a code base that provides end-to-end training and evaluation with  unit tests. Anticipating reproduction we wanted to report on the training of the model and understand what it was learning. Unit tests greatly helped in our development process in reproducing the paper. When faced with an issue, tests ensured that certain components in our code bases were working as expected. We modeled our implementation off of [nussl](https://github.com/nussl/nussl) and pytorch supported libraries like torchvision.

To help with compartmentalizing the code, We built a set of [transforms classes](https://github.com/blainerothrock/hyperspectral-imaging-ml/tree/master/hyperspec/transforms) that handle the pre-processing of the hyperspectral image and unit tested each individually with toy data. Following the paper’s implementation we ran PCA with whitening (from scikit-learn) over the entire image with a _K_ value of 30. Then we split the data into smaller images using a window size of 25. This logic is performed as part of the [Indian Pine Dataset](https://github.com/blainerothrock/hyperspectral-imaging-ml/blob/master/hyperspec/datasets/indian_pine.py) (Pytorch) subclass and is abstracted from the training code. Hyperparameters are configurable from the experiments gin file.

Splitting the images into cubes is arguably the most complicated code block and it is not very efficient. While we wrote unit tests to ensure the functionality, the method itself is taken from the paper's code base. Hyperspectral images are large and having an inefficient method for splitting this data into cubes results in a limited window size. If given more time, reconstructing this method so that smaller windows are possible would be interesting to compare against the papers window size of 25.

**Model**

We implemented the [model](https://github.com/blainerothrock/hyperspectral-imaging-ml/blob/master/hyperspec/model/hybridsn.py) using a single PyTorch model class. We ran into some difficulties with dimensionality that is discussed on our tips and tickets document. Overall the model is relatively simple, but did require some hard-coding of dimensions. Specifically when resizing from the 3D convolution to 2D. We were able to compare our model using the output of the Keras model in the provided jupyter notebook. Minus some differences in internal dimensions, the models are identical and match in the number of parameters with a windows size of 25 and K of 30 which is 5,122,176 trainable parameters. Dropout of `0.4` is used between the linear output layers. For reference each sample contains 18,750 data points. Below is a summary of the model:
```text
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv3d-1        [-1, 8, 23, 23, 24]             512
              ReLU-2        [-1, 8, 23, 23, 24]               0
            Conv3d-3       [-1, 16, 21, 21, 20]           5,776
              ReLU-4       [-1, 16, 21, 21, 20]               0
            Conv3d-5       [-1, 32, 19, 19, 18]          13,856
              ReLU-6       [-1, 32, 19, 19, 18]               0
            Conv2d-7           [-1, 64, 17, 17]         331,840
              ReLU-8           [-1, 64, 17, 17]               0
            Linear-9                  [-1, 256]       4,735,232
             ReLU-10                  [-1, 256]               0
          Dropout-11                  [-1, 256]               0
           Linear-12                  [-1, 128]          32,896
             ReLU-13                  [-1, 128]               0
          Dropout-14                  [-1, 128]               0
           Linear-15                   [-1, 16]           2,064
================================================================
Total params: 5,122,176
Trainable params: 5,122,176
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.07
Forward/backward pass size (MB): 7.17
Params size (MB): 19.54
Estimated Total Size (MB): 26.78
----------------------------------------------------------------
```

**Training**

In order to train the model, we used Torch Ignite to simplify our implementation and allow easy access to reporting and saving the model. We split the data 70/30 as defined in the paper’s code base, resulting in 7,175 training samples and 3,074 testing. While training we recorded training and testing loss and accuracy at each epoch. We also captured loss at each iteration step simply for debugging purposes. We used cross entropy for the loss which computes the probability across all classes. For optimization we used the Adam optimizer with a learning rate of `0.001` and weight decay `10<sup>-6</sup>`. Batch size was set at `256`. All hyperparameters matched that of the HybridSN paper. These hyper parameters are managed using gin-config file. We trained for 100 epochs. View the [readme][README.md] for details on how to reproduce the training.

**Results**

Although we ran the model for 100 epochs, it consistently converged in under 20. This roughly matched the paper’s report of ~15 minutes training on a consumer GPU. The original results were reported on a GTX1060 and we used a GTX1080 so the results should be similar if not slightly faster. Our results yielded accuracy of 100% on the training data and just below 100% on testing. The oringally results are offically report at 99.81% accuracy. not that .19% corresponds to less than 20 miss-classified examples. 

![results_img](https://blainerothrock-public.s3.us-east-2.amazonaws.com/img/HybridSN_training_results.png)

Our initial assumption is that the model overfits and given the similarity in the data is able to identify over the testing set. Each example contains 18,750 data points and with 7,175 training samples there are 134,531,250 data points to be learned with 5,122,176 trainable parameters. There is definitely room for generalization given that simple analysis, but perhaps the redundancy and physics behind hyperspectral imaging results in simple feature extraction. In continuing to explore this model we would first like to attempt to simplify the model and identify where the results start to hinder. A better test, we assume, would be to try this model out using different datasets that provide more examples and more classes to see if the current parameters yield sufficient results. 