# Emotion Detection Using AlexNet

## Overview

This project implements **facial emotion detection** using the AlexNet Convolutional Neural Network (CNN) architecture. The model is trained on grayscale facial images to classify emotions into seven categories: **Anger, Disgust, Fear, Happiness, Neutral, Sadness, and Surprise**.

The goal is to build an accurate and efficient emotion recognition system that can be applied to images or real-time video feeds.

## Features

- Emotion classification into 7 categories using AlexNet CNN
- Preprocessing of grayscale facial images resized to AlexNet input dimensions (128x128)
- Data augmentation to improve model generalization
- Training from scratch.
- Web application using Streamlit.


## Dataset

The model is trained on a publicly available facial emotion dataset, such as the **FER2013**  datasets containing  35,685  labeled 48x48 grayscale facial images covering the seven basic emotions.

## Model Architecture

- Based on the original **AlexNet** architecture with:
    - 5 convolutional layers
    - Max-pooling and ReLU activations
    - Batch normalization and dropout for regularization
    - Fully connected layers adapted for 7-class classification
- Input images resized and replicated across 3 channels if required


## Results

- Achieved approximately **90-95% accuracy** on the test set depending on training parameters and dataset used.
- Model performance can be improved with data augmentation and hyperparameter tuning.


## Challenges

- Training AlexNet from scratch requires substantial computational resources.
- Large model size (~800MB) may require separate download or optimization.


## References

- AlexNet: Krizhevsky et al., 2012, ImageNet Classification with Deep Convolutional Neural Networks
- FER2013 Dataset: Kaggle - Facial Expression Recognition Challenge

## Sample Web Application Screenshots


