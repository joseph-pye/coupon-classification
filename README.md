# coupon-classification
This simple project produces two logistic regression models from the in vehicle coupon classification dataset from the UCI Machine learning repository. The first model is a simple logistic regression using the input variables after being transformed appropriately using dummy variables and scaling. The second model uses all of the second order interaction terms, to capture the interactions between variables such as "Is the coupon for a bar?" and "How many times do you visit a bar in 1 month?". These are obviously important interactions, which is reflected in the accuracy of the model. Unfortunately, it's likely that this also introduces overfitting, which would need to be addressed if the model was going to be developed further.

Running the code produces the following outputs:
-A list of variables in the dataset along with the values of that variable and the number of occurrences.
-A confusion matrix and accuracy score for the first order model
-A confusion matrix and accuracy score for the second order model

The code uses python 3.9