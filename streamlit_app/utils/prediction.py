import numpy as np
import pandas as pd

def make_prediction(model, year, month):
    """
    Use year and month to make temperature prediction
    """
    input_features = np.array([[year, month]])

    return model.predict(input_features)[0]