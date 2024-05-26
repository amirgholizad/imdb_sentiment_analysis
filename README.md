# IMDB Reviews Sentiment Analysis

## Project Overview

This project involves performing sentiment analysis on IMDB movie reviews. The goal is to classify the sentiment of each review as either positive or negative. By leveraging machine learning techniques, specifically a neural network model, we aim to achieve a high level of accuracy in this classification task.

## Model Implementation

To tackle this problem, we implemented a Sequential model using Keras, a high-level neural networks API. The architecture of the model consists of the following components:

1. **Input Layer**: This layer takes the tokenized and preprocessed text data as input.
2. **Hidden Layers**: We used three Dense (fully connected) layers. Each layer is followed by a Sigmoid activation function, which helps in capturing the non-linearity in the data.
3. **Output Layer**: The final layer is also a Dense layer with a Sigmoid activation function, outputting a probability score that indicates whether a review is positive or negative.

## Model Compilation

The model was compiled with the following parameters:
- **Loss Function**: `categorical_crossentropy`, which is suitable for classification tasks where the output is one of two or more classes.
- **Optimizer**: `Adam`, an efficient gradient-based optimization algorithm that adapts the learning rate during training.

## Training and Evaluation

The model was trained on a dataset of IMDB reviews, which was split into training and testing sets. After training, the model was evaluated on the testing set, achieving an impressive 85% precision in classifying the reviews into "positive" and "negative" groups.

## Key Results

- **Model Architecture**: Sequential model with 3 Dense layers and Sigmoid activation functions.
- **Loss Function**: Categorical Cross-Entropy
- **Optimizer**: Adam
- **Precision**: 85% on the testing set

## Conclusion

The sentiment analysis model successfully classifies IMDB reviews with high precision. This demonstrates the effectiveness of neural network models in natural language processing tasks, specifically sentiment analysis.