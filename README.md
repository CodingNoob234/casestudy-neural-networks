# Case Study - Topic: Neural Networks
### Joey Besseling, Siem Fenne, Mees Petrus

## Introduction
In this project, the aim is to model the daily volatility of stocks and crypto assets using neural networks, and compare their performance against the well-performing HAR model from the literature.

## Methods
The main.ipynb notebook file is build up in a list of steps:

### Data Preparation
- The data daily returns are fetched through the Yahoo Finance API, or through importing a csv file.
- The daily volatility (and weekly/monthly directly resulting from this) are computed
- This data is used as features and targets for BOTH models! So both have the same output and we can observe if the neural network is able to capture more complex patterns that lead to better predictive power.
- This data is transformed into training and validation data. The validation data is only used in the end to compare the performance of our final model and the HAR model.

### Model Specification
- The training set is split into several training and validation sets by TimeSeriesKfold to apply cross validation.
- A list of possible model specifications is used; like number of hidden layers and the number of nodes in each layer, learning rate and number of epochs.
- For each kfold, the model is estimated using MSELoss and Adam's version of gradient descent. The number of epochs is fixed to 10, with early stopping options if the performance does not increase.
- The model specification with the best MSELoss over all kfolds is chosen as the final model specification.

### Final Model Estimation and Comparison
- According to the determined final model specification, the model is estimated on the complete training data.
- The HAR model is estimated on exactly the same training data.
- Both models predict the testing data and are compared by MSE.

## Code Improvements
- A big problem that seems to arise is the inconsistency of the trained neural network. The occurence of this problem is well known and the torch documentation provides some explanation how to tackle this partially.
- Generalize more functions.
- Look if there is a torch model builder so you can pass the number of layers etc. as variables to a function/class that automatically initializes the model and forward function.
- Some of the data processing can be standardized into functions.