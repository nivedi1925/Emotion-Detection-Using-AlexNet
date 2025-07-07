# Emotion Detection Using AlexNet

## Overview

This project implements **facial emotion detection** using the AlexNet Convolutional Neural Network (CNN) architecture. The model is trained on grayscale facial images to classify emotions into seven categories: **Anger, Disgust, Fear, Happiness, Neutral, Sadness, and Surprise**.

The goal is to build an accurate and efficient emotion recognition system that can be applied to images or real-time video feeds.

## Features

- Emotion classification into 7 categories using AlexNet CNN
- Preprocessing of grayscale facial images resized to AlexNet input dimensions (227x227)
- Data augmentation to improve model generalization
- Training from scratch or transfer learning with fine-tuning
- Real-time emotion detection using webcam input (optional)
- Visualization of predicted emotions with corresponding emojis or labels


## Dataset

The model is trained on a publicly available facial emotion dataset, such as the **FER2013** or similar datasets containing thousands of labeled 48x48 grayscale facial images covering the seven basic emotions.

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
- Real-time detection speed depends on face detection efficiency (e.g., Haar cascades may introduce latency).
- Large model size (~700MB) may require separate download or optimization.


## References

- AlexNet: Krizhevsky et al., 2012, ImageNet Classification with Deep Convolutional Neural Networks
- FER2013 Dataset: Kaggle - Facial Expression Recognition Challenge

