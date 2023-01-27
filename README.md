# Case Study - Topic: Neural Networks
### Joey Besseling, Siem Fenne, Mees Petrus

## Introduction
In this project, the aim is to model the daily volatility of stocks and crypto assets using neural networks, and compare their performance against the well-performing HAR model from the literature.

## Methods
The main.ipynb notebook file is build up in a list of steps explained below.

### Data Preparation
- The data daily returns are fetched through the Yahoo Finance API, or through importing a csv file.
- The daily volatility (and weekly/monthly directly resulting from this) are computed
- This data is used as features and targets for BOTH models! So both have the same output and we can observe if the neural network is able to capture more complex patterns that lead to better predictive power.
- This data is split into training and validation data. The validation data is only used in the end to compare the performance of our final neural network and the HAR model.

### Model Specification
- The training set is split into several training and validation sets by TimeSeriesKfold to apply cross validation.
- A list of possible model specifications is used; like number of hidden layers and the number of nodes in each layer, learning rate and number of epochs.
- For each kfold, the model is estimated using MSELoss and Adam's version of gradient descent. The number of epochs is fixed to 10, with early stopping options if the performance does not increase.
- The model specification with the best MSELoss over all kfolds is chosen as the final model specification.

### Final Model Estimation and Comparison
- According to the determined final model specification, the model is estimated on the complete training data.
- The HAR model is estimated on exactly the same training data.
- Both models predict the testing data and are compared by MSE.

========================================================================================================================
## Code Improvements
- Add in sample fitting, and plot the activation function per node with a range of parameter values as input. Through this, we can observe what 'activate' the node.
- The realized volatility from the adviced website seems to have negligable correlations for lagged values. The models, both HAR and NN, thus don't capture any relation in the data.
- The dataloaders weren't balanced before, now set to True.

## Q&A's
- How can we extract the 5-minute high frequency data from the WRDS database.
- Currently we are using the same amount of features for both models. Can we include lagged versions of the features for the neural networks. If so, how do we determine the amount of lags?
- How can we find the optimal seed (parameter initialization) to train the neural network? Should the seed be included when cross validating?
- Is there a way to find the best combination of hidden layers and nodes, apart from a GridSearch?
- Are we supposed to perform the goodness-of-fit for the models (MSE, likelihood, R^2)?

## Feedback Enzo
- Diebold-Moriano for comparing accuracy -> added
- Trade off complexity capturing highly non-linear relations, versus simplicity, estimation consistency and faster.
- 2 Nodes (only option between input output) -> this is fixed
- Activation function is IMPORTANT! Especially for insample estimation and analysis --> changed to sigmoid function
- Seed fixed first, then random (want stochatisticity in the model)
- More input (like lagged values) is something that can be added as extra in the report, no prio