import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score


def load_data(file):
    # Take csv file and read it into a pandas DataFrame
    data = pd.read_csv(file)
    return (data)


def explore_data(data):
    # Print some summary metrics about the data. This is aimed towards categorical data.
    print(data.head)
    print(data.dtypes)
    for col in data.columns:
        print(col)
        print(pd.value_counts(data[col]))


def prep_data(data):
    # Convert categorical variables to dummy variables, scale numeric variables and split into X and Y
    data = pd.get_dummies(data)

    sc = StandardScaler()
    data.loc[:, 'temperature'] = sc.fit_transform(data[['temperature']])

    X = data.loc[:, data.columns.values != 'Y'].copy()
    y = data.loc[:, data.columns.values == 'Y'].copy()

    return X, y


def split_data(X, y, test_size=0.1, random_state=0):
    # Split data into a training set and a testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


def fit_lr(X_train, y_train, max_iter=1000):
    # Fit a logistic regression model to the training set
    lr = LogisticRegression(max_iter=max_iter)
    lr.fit(X_train, np.ravel(y_train))

    return lr


def evaluate_lr(lr, X_test, y_test):
    # Make predictions with the model and produce some evaluation metrics
    y_pred = lr.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    return cm, acc


def add_interactions(X):
    # Add all the second order interaction terms to input variables
    pn = PolynomialFeatures()
    X2 = pn.fit_transform(X)
    return X2


# This is the filename of the input data
file = "in-vehicle-coupon-recommendation.csv"
# Load the data
data = load_data(file)
# Print some summary metrics about the data to give us an idea what it looks like
explore_data(data)
# Convert categorical variables to dummy variables, scale numeric variables and split into X and Y
X, y = prep_data(data)
# Split into a training set and a testing set
X_train, X_test, y_train, y_test = split_data(X, y)
# Fit the model to the vanilla training data
lr = fit_lr(X_train, y_train)
# Capture some evaluation metrics for the model
cm, acc = evaluate_lr(lr, X_test, y_test)
# There are some interactions that are obviously important, for example the coupon type
# with the use of different venue types. Here I create a new input dataframe with all second
# order interaction terms. In future this can be done more selectively and elegantly.
X2 = add_interactions(X)
# Repeat the process with the new inputs
X2_train, X2_test, y_train, y_test = split_data(X2, y)
lr2 = fit_lr(X2_train, y_train)
cm2, acc2 = evaluate_lr(lr2, X2_test, y_test)
# Print out the evaluation metrics to compare before and after adding interaction terms.
print('Summary metrics for 1st order features: ')
print(cm)
print(acc)
print('Summary metrics for 2nd order interaction terms: ')
print(cm2)
print(acc2)
