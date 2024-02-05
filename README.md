# Case Study - Topic: Neural Networks
### Joey Besseling, Siem Fenne, Mees Petrus

## Introduction
In this project, the aim is to model the daily volatility of certain stocks using neural networks, and compare their performance against the well-performing HAR model from the literature.

## Methods
The main.ipynb notebook contains a list of steps, explained below.

### Data Preparation
- The daily returns are fetched through the Yahoo Finance API, or through importing a csv file (from CRISPR).
- The daily volatility (and weekly/monthly directly resulting from this) are computed
- This data is used as features and targets for BOTH models! So both have the same input and we can observe if the neural network is able to capture more complex patterns that lead to more predictive power.
- This data is split into training and validation data. The validation data is only used in the end to compare the performance of our final neural network and the HAR model.

### Model Specification
- The training set is split into several training and validation sets by TimeSeriesKfold to perform cross validation.
- A list of possible model specifications is used; like number of hidden layers and the number of nodes in each layer, learning rate and number of epochs.
- For each fold, the model optimizes MSELoss with Adam's version of gradient descent. The number of epochs is fixed to 10, with early stopping options if the performance does not increase further.
- The model specification with the best MSELoss over all folds is chosen as the final model specification.

### Final Model Estimation and Comparison
- According to the determined final model specification, the model is trained on the complete training data.
- The HAR model is estimated on exactly the same training data (HAR has no hyperparameters, hence not included in the CV).
- Both models predict the testing data (1 day ahead) and are compared by MSE.
