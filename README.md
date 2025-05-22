# Animal Image Classification

## Project Overview

This project focuses on developing and comparing various deep learning models for animal image classification. The goal is to accurately classify images into different animal categories using computer vision techniques and deep learning architectures.

## Dataset

The project uses an animal image dataset containing multiple animal categories. The dataset is organized into folders, with each folder corresponding to a specific animal class.

Key dataset characteristics:

- Multiple animal categories (e.g., african_elephant and others)
- Images stored in JPG format
- Dataset is preprocessed through filtering to remove problematic images

## Project Structure

The project is organized into several Jupyter notebooks, each handling a different aspect of the machine learning pipeline:

1. **Project_1_Data_Preparation.ipynb**: Handles data downloading, extraction, and cleaning
2. **Project 1*CV* Animals classification.ipynb**: Contains the main classification pipeline including data preparation and initial model training
3. **Project_1_Experiment_1.ipynb**: First experiment focusing on basic classification models
4. **Project_1_Experiment_2.ipynb**: Second experiment investigating the impact of transfer learning
5. **EfficientNet_experiment.ipynb**: Experiment using EfficientNet architecture as an alternative to VGG16
6. **CNN-ensembles_experiment.ipynb**: Experiment combining Gabor filtering with CNN, ViT, and InceptionNet models
7. **augmented_code.ipynb**: Contains code for data augmentation to enhance the training dataset

## Methodologies

### Data Preprocessing

- Data cleaning using SVM models to automatically detect and filter out problematic images
- Image standardization and normalization
- Data augmentation (rotation, width/height shifts, shear, zoom, horizontal flip) to enhance training data

### Model Architectures

The project explores multiple deep learning architectures:

1. **Support Vector Machine (SVM)**: Used for initial data filtering and basic classification
2. **VGG16**: Transfer learning implementation
3. **EfficientNet**: More efficient alternative to VGG16
4. **CNN Ensembles**: Combining multiple CNN architectures
5. **Vision Transformer (ViT)**: Transformer-based approach to image classification
6. **InceptionNet**: Google's inception architecture

### Transfer Learning

The project investigates the impact of transfer learning by utilizing pre-trained models and fine-tuning them on the animal dataset.

## Requirements

- Python 3.x
- TensorFlow/Keras
- scikit-learn
- scikit-image
- numpy
- matplotlib
- PIL (Python Imaging Library)
- imageio

## Usage

### Data Download and Preparation

```python
# Download and extract dataset
!gdown --id 19Rr-b09YUjcmgc6_mJyTZHSOsr-ZPNNP
!unzip -qq /content/animals.zip
```

### Data Augmentation

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
        rotation_range=45,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='reflect')

# Apply augmentation to images
# See augmented_code.ipynb for full implementation
```

### Training Models

Refer to the specific experiment notebooks for detailed model training procedures:

- Project_1_Experiment_1.ipynb
- Project_1_Experiment_2.ipynb
- EfficientNet_experiment.ipynb
- CNN-ensembles_experiment.ipynb

## Experiments and Results

The project includes several experiments:

1. **Basic Classification**: Initial models to establish baseline performance
2. **Transfer Learning**: Evaluation of the impact of using pre-trained models
3. **EfficientNet vs. VGG16**: Comparison of performance and efficiency
4. **Ensemble Methods**: Investigation into combined models (Gabor filtering + CNN + ViT + InceptionNet)

## Future Work

- Incorporate additional animal categories
- Experiment with additional state-of-the-art architectures
- Optimize models for deployment on edge devices
- Explore explainable AI techniques to understand model predictions

## References

- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
- [Very Deep Convolutional Networks for Large-Scale Image Recognition (VGG16)](https://arxiv.org/abs/1409.1556)
- [Going deeper with convolutions (Inception/GoogLeNet)](https://arxiv.org/abs/1409.4842)
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT)](https://arxiv.org/abs/2010.11929)
