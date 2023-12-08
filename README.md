# Charity Funding Analysis with Tensor and Neural Networks

## Overview of the Analysis

The purpose of this analysis is to create a binary classification model using deep learning techniques to predict if a particular charity organization will succeed in selecting successful applicants for funding. The model draws on a dataset of over 34,000 organizations that have received funding in the past.

## Results

### Data Preprocessing

- The variable, IS_SUCCESSFUL, is the target of the binary classification model.

- The variables – APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, and ASK_AMT – are the features of the model.

- The variables, EIN and NAME, are neither targets nor features and are not part of the input data.

### Compiling, Training, and Evaluating the Model

- The model has two hidden layers and one output layer.  The two hidden layers consist of 95 and 38 neurons, respectively, and use ReLU activation functions.  Because this is a binary classification model, the output layer has 1 neuron and uses a Sigmoid activation function.  The structure maintains the ability to learn patterns effectively while striking a balance between complexity and overfitting.

<img width="652" alt="Screenshot 2023-12-07 at 9 16 52 PM" src="https://github.com/njgeorge000158/deep-learning-challenge/assets/137228821/e6089cbf-688d-4306-ad68-859e34a6f21b">

