import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def data_split(X, y, test_size = 0.3):
    """
    split the data into training and testing
    """
    return train_test_split(X, y, test_size = test_size, random_state = 42)

def train(X_train, y_train, model_type = "Linear Regression"):
    """
    model training based on specified algorithm
    """
    if model_type == "Linear Regression":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators = 150, random_state = 42)

    model.fit(X_train, y_train)

    return model

def model_evaluation(model, X_train, y_train, X_test, y_test):

    y_pred_train = model.predict(X_train) 
    y_pred_test = model.predict(X_test)
    metrics = {
        'training_rmse' : np.sqrt(mean_squared_error(X_train, y_train)),
        'test_rmse' : np.sqrt(mean_squared_error(X_test, y_test)),
        'training_r2' : r2_score(X_train, y_train),
        'test_r2' : r2_score(X_test, y_test)
        'y_test': y_test,
        'y_pred': y_pred
    }