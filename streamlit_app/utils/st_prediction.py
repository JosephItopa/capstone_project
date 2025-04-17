import streamlit as st
import numpy as np

from .prediction import make_prediction
from .model_training import load_model


def show_pred():

    # prediction input
    st.subheader("select values for prediction")
    pred_year = st.slider("year", 2020, 2040, 2025)
    pred_month = st.slider("month", 1, 12, 6)

    # load model
    model = load_model()

    # make prediction
    predicted = make_prediction(model, pred_year, pred_month)

    # Display the results
    st.success(f"Predicted temperature for {pred_year} - {pred_month}: {predicted:.2f} in Celcius")