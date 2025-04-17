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

def model_evaluation(model, X_train, X_test, y_train, y_test):
    y_pred_train = model.predict(X_train) 
    y_pred_test = model.predict(X_test)
    metrics = {
        'training_rmse' : np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'test_rmse' : np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'training_r2' : r2_score(y_train, y_pred_train),
        'test_r2' : r2_score(y_test, y_pred_test),
        'y_test': y_test,
        'y_pred': y_pred_test
    }

    return metrics

def save_model(model, file_name = "nepal_climate_model.pkl"):
    """
    to save model as pickle file
    """
    with open(file_name, 'wb') as file:
        pickle.dump(model, file)

def load_model(file_name = "nepal_climate_model.pkl"):
    """
    to load the saved model
    """
    try:
        with open(file_name, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        return None
    