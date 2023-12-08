# Charity Funding Analysis with Tensor and Neural Networks

## Overview of the Analysis

The purpose of this analysis is to create a binary classification model using deep learning techniques to predict if a particular charity organization, Alphabet Soup, will succeed in selecting successful applicants for funding. The model draws on a dataset of over 34,000 organizations that have received funding in the past.

## Results

### Data Preprocessing

- The variable, IS_SUCCESSFUL, is the target of the binary classification model.

- The variables – APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, and ASK_AMT – are the features of the model.

- The variables, EIN and NAME, are neither targets nor features and are not part of the input data.

### Compiling, Training, and Evaluating the Model

- The model has two hidden layers and one output layer.  The two hidden layers consist of 95 and 38 neurons, respectively, and use ReLU activation functions.  Because this is a binary classification model, the output layer has 1 neuron and uses a Sigmoid activation function.  The structure maintains the ability to learn patterns effectively while striking a balance between complexity and overfitting.

<img width="652" alt="Screenshot 2023-12-07 at 9 16 52 PM" src="https://github.com/njgeorge000158/deep-learning-challenge/assets/137228821/e6089cbf-688d-4306-ad68-859e34a6f21b">

- Unfortunately, this model did not achieve the target model performance of at least 75% predictive accuracy.

<img width="767" alt="Screenshot 2023-12-07 at 9 21 00 PM" src="https://github.com/njgeorge000158/deep-learning-challenge/assets/137228821/44831d4b-325f-43fd-9776-fccd87102b6e">

To achieve the target performance, I made numerous changes to the data set, preprocessing, and neural network configuration. First, I dropped the EIN, STATUS, SPECIAL_CONSIDERATIONS columns from the data set: these columns either had to many uniquely distributed values or virtually none.  Next, I wrote a neural network optimization program, AlphabetSoupCharityOptimizationSearch.ipynb, that specified the following cutoff values for the variables, NAME, APPLICATION_TYPE, CLASSIFICATION, USE_CASE, and ASK_AMT: 2, 157, 96, and 4.  According to the program results, the optimal model for this data set contained four hidden layers with 58, 19, 97, 48, and 7 neurons, respectively, using leaky ReLU activation.

<img width="585" alt="Screenshot 2023-12-07 at 9 33 24 PM" src="https://github.com/njgeorge000158/deep-learning-challenge/assets/137228821/5eb7123c-b22c-4495-85be-30e5494bf2a7">

Once implemented, the optimized model attained a pedictive accuracy of 80.87%.

<img width="751" alt="Screenshot 2023-12-07 at 9 35 39 PM" src="https://github.com/njgeorge000158/deep-learning-challenge/assets/137228821/94d7b58c-7f89-4028-a692-86eaf5fd2803">

## Summary

Overall, through optimization, the model exceeded the target predictive accuracy of 75% with 80.87%.  If I were to attempt to improve performance in the future, I would modify the optimization program to include other neural network configurations beyond sequential.



