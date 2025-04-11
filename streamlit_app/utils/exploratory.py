import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_time_series(df):
    """
    plot yearly average temperature
    """
    fig, ax = plt.subplots(figsize = (10, 6))
    yearly_average = df.groupby('year')['temperature'].mean().reset_index()
    ax.plot(yearly_average['year'], yearly_average['temperature'], marker = '*')
    ax.set_xlabel("year")
    ax.set_ylabel("Average Temperature in Celcius")
    ax.set_title("Yearly Average Temperature")
    ax.grid(True)
    return fig

def plot_seasonality(df):
    """
    plot monthly temperature trend
    """
    fig, ax = plt.subplots(figsize = (10, 6))
    sns.boxplot(x = "month", y = "temperature", data = df, ax = ax)
    ax.set_xlabel("Month")
    ax.set_ylabel("Average Temperature in Celcius")
    ax.set_title("Monthly Average Temperature Distribution")
    return fig

def plot_yearly_trend(df):
    """
    plot of r yearly trend
    """
    fig, ax = plt.subplots(figsize = (10, 6))
    yearly_trend = df.groupby('year')['temperature'].mean().reset_index()
    ax.plot(yearly_trend['year'], yearly_trend['temperature'], marker = 'o')
    ax.set_xlabel("Year")
    ax.set_ylabel("Average Temperature in Celcius")
    ax.set_title("Yearly Average Temperature")
    return fig

def plot_actual_vs_predicted(y_test, ypred):
    return