import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from utils.preprocess import load_data
from utils.st_exploratory import show_analysis

# root__________
#       |--------app.py
#       |________utils
#           |-------preprocess.py
#           |-------exploratory.py
#           |-------st.exploratory.py
#           |-------model_training.py
#           |-------st.model_training.py

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

if page == "EDA":
    # call a responsible for EDA
    show_analysis(df)
elif page == "Model Training":
    # algorithm training
    print("")
else:
    # run prediction
    print("")


# df = load_data()
# st.dataframe(df)

