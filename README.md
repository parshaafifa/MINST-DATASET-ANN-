# MNIST Digit Classification with TensorFlow/Keras

This repository contains a simple neural network model to classify handwritten digits from the MNIST dataset using TensorFlow and Keras.

---

## Overview

The MNIST dataset includes 70,000 grayscale images of handwritten digits (0-9), each sized 28x28 pixels. This project demonstrates:

- Loading and preprocessing the MNIST dataset
- Building a feedforward neural network with dropout regularization
- Training the model and visualizing training accuracy
- Evaluating model performance on the test set
- Predicting and displaying sample digits with their labels

---

## Workflow Description

This project follows a straightforward workflow to build, train, and evaluate a neural network model for MNIST digit classification. The steps are:

1. **Import Libraries**  
   Load essential Python libraries for numerical computing, data handling, visualization, image processing, and deep learning.

2. **Load Dataset**  
   Use Keras’ built-in MNIST dataset loader to download and load the MNIST handwritten digit images and their corresponding labels into training and test sets.

3. **Explore Data**  
   Inspect dataset shapes, types, and display sample images with their labels to understand the data format and distribution.

4. **Preprocess Data**  
   Normalize image pixel values from [0,255] to [0,1] to improve neural network training.

5. **Build the Model**  
   Define a Sequential neural network with flatten, dense, and dropout layers, culminating in a softmax output layer for classification.

6. **Compile the Model**  
   Configure the optimizer, loss function, and evaluation metrics.

7. **Train the Model**  
   Fit the model on training data with validation split to monitor performance over epochs.

8. **Visualize Training**  
   Plot training and validation accuracy to assess learning progress.

9. **Evaluate Model**  
   Predict on test data and compute accuracy score.

10. **Make Predictions**  
    Use the trained model to predict digit classes for individual sample images.

---

## Getting Started

### Prerequisites

Make sure you have Python 3.x installed. Install required packages:

```bash
pip install numpy pandas matplotlib seaborn tensorflow keras scikit-learn pillow opencv-python
