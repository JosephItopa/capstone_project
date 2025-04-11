import streamlit as st
from .exploratory import plot_seasonality, plot_time_series, plot_yearly_trend

def show_analysis(df):
    """
    Display exploratory data analysis
    """
    st.header("Exploratory Data Analysis")

    # show raw data
    st.subheader("Raw Temperature Data")
    st.dataframe(df.sample(10))

    # basic statistics
    st.subheader("Descriptive Analysis") # Statistical summary
    st.write(df["temperature"].describe())

    # Plot the time series
    st.subheader("Temperature over time")
    fig = plot_time_series(df)
    st.pyplot(fig)

    # Plot seasonality
    st.subheader("Seasonal Temperature Trend")
    fig = plot_seasonality(df)
    st.pyplot(fig)

    # Yearly Average Temperature
    st.subheader("Yearly Average Temperature")
    fig = plot_yearly_trend(df)
    st.pyplot(fig)

