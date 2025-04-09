import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from preprocess import load_data 


# set page configuration
st.set_page_config(
    page_title = "climate trend predictor",
    layout="wide"
)

# App title and description
st.title("Climate Trend Analysis & Predictor")

st.markdown("Analyze historical temperature and predict trend")

# sidebar 
st.sidebar.title("Navigation Page")
page = st.sidebar.radio("Go to", ["EDA", "Model Training", "Prediction"])

df = load_data()

st.dataframe(df)

