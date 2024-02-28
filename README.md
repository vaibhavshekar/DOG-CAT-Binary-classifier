The dataset consists of images of cats and dogs, sourced from various online repositories.

The dataset underwent preprocessing steps to ensure data integrity and consistency. Initially, images were checked for valid formats and extensions. Any corrupted or unsupported files were removed from the dataset. Subsequently, the images were loaded using OpenCV and normalized to scale pixel values between 0 and 1. This normalization step ensures uniformity and aids in model convergence during training.

The CNN model architecture consists of three convolutional layers followed by max-pooling layers to extract spatial features from the images. The convolutional layers employ ReLU activation functions to introduce non-linearity and facilitate feature learning. The output of the convolutional layers is flattened and passed through fully connected layers with ReLU activation functions. Regularization techniques such as L1 and L2 regularization were applied to mitigate overfitting.

# model.add(BatchNormalization()) #tuning - reduced precision and accuracy significantly when Batch Normalization was applied, a significant drop in prediction and accuracy validation was observed .

dataset: https://www.kaggle.com/datasets/karakaggle/kaggle-cat-vs-dog-dataset
