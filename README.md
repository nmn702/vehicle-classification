# Vehicle Classification using ResNet18

A deep learning project that uses a **pretrained ResNet18** model to classify vehicle images into **7 different classes** using PyTorch.

## Overview

The project:

* Loads a vehicle image dataset using `ImageFolder`.
* Resizes images to `224 × 224`.
* Splits the dataset into **80% training** and **20% testing** data.
* Uses a pretrained **ResNet18** model.
* Replaces the final layer with a 7-class classifier.
* Trains the model using **Cross-Entropy Loss** and **Adam optimizer**.
* Evaluates the model on the test set.
* Displays sample predictions for each vehicle class.

## Model

**Architecture:** ResNet18
**Framework:** PyTorch
**Input Size:** `224 × 224`
**Number of Classes:** `7`
**Epochs:** `3`
**Batch Size:** `32`
**Learning Rate:** `0.001`

```python
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 7)
```

## Dataset Structure

The dataset is expected to follow the `ImageFolder` structure:

```text
Vehicles/
├── class_1/
├── class_2/
├── class_3/
├── class_4/
├── class_5/
├── class_6/
└── class_7/
```

## Requirements

* Python 3.x
* PyTorch
* Torchvision
* NumPy
* Matplotlib
* Pillow

## Running

The notebook is designed to run in **Google Colab** and loads the dataset from Google Drive.

```bash
python mlbasictask.py
```

## Concepts Used

* Image Classification
* Transfer Learning
* Convolutional Neural Networks
* ResNet18
* PyTorch
* GPU Acceleration
* Model Evaluation

> **Note:** This project is intended for educational purposes and demonstrates a basic transfer-learning workflow for image classification.
