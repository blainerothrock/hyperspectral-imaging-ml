# hyperspectral-imaging-ml

## Papers:
* [Deep Learning for Classification
of Hyperspectral Data: A Comparative Review](https://arxiv.org/pdf/1904.10674.pdf)
    - An overview of the field relating to deep learning
    - [code base](https://github.com/nshaud/DeepHyperX)
* [HybridSN: Exploring 3D-2D CNN Feature Hierarchy for Hyperspectral Image Classification](https://arxiv.org/pdf/1902.06701v3.pdf)
  - Current state-of-the-art on the Indian Pines, Pavia University and Salinas Scene datasets
  - [code base](https://github.com/gokriznastic/HybridSN)
* [Hyperspectral Image Classification with Deep Metric Learning and Conditional Random Field](https://arxiv.org/pdf/1903.06258v2.pdf)
  - State of the art without additional data on the Indian Pines data set
  -  None :(

## Datasets
* [Overview](http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes)
* [Indian Pine](https://purr.purdue.edu/publications/1947/1)
* [Data Fusion Contest 2018](https://mediatum.ub.tum.de/1474000?id=1474000)

## Why
Hyper-spectral imaging is a upcoming field that [has potential](https://www.cloudagronomics.com/technology) in the agriculture industry with many benefits including crop yield and carbon monitoring.

## Paper Review
* Rigor vs. Empirical - Balanced?
* Readability - Excellent
* Algorithm Difficulty - Low
* Pseudo Code - None / Step-Code?
* Hyperparameters Specified - Yes
* Compute Needed - GPU
* Number of Equations - 2
* Number of Tables - 5

## Paper Notes
* Proposes a hybrid 3d and 2d model for general hyperspectral image(HSI) classification
* 3-D CNN: Employs principal component analysis on input data to reduce spatio-spectral images by its spectral bands(depth) in order to remove spatial redundancy
  - 3D convolution → 3D kernel convolves on 3D-data(spatio-spectral image)
  - Uses 3d patches to determine image classification
  - 3D patches: overlapping spatio-spectral convolutions where the centered pixel is used for classification
  - Computationally expensive
  - Papers recommend 3 layered model to extract spectral features
    - One paper dubs this the Deep Metric Learning followed by a Conditional Random Field layer to make predictions
* 2-D CNN: Input data is convolved with 2d kernels(normal)
* Hybrid of both 3D and 2D Kernels are used for learning
  - Use of 3D convolutions to capture spatial data and 2D convolutions to decrease computational expense and learn non-spectral information (features of images for classification)
* Utilizes both spatio-spectral imaging in the form of 3-d convulsions and non spatio-spectral imaging in the form 2d convolutions
* This model also shows great performance with little data


Conclusion: We believe the paper is highly reproducible and very well documented. The only potential issue we foresee is within the preprocessing phase.
