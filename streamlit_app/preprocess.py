import pandas as pd
import numpy as np

def load_data():
    # generate data
    # create data for years of monthly temperature average
    date_array = pd.date_range(start='2015-03-01', end='2025-03-31', freq='M')

    # synthetic temperature data with seasonal patterns 3.142
    temps = []
    for i in range(len(date_array)):
        # base temperature with seasonal pattern
        seasonal = 20 + 10 * np.sin(2*np.pi*i/12)
        trend = 0.03 * i
        noise = np.random.normal(0, 1.5)
        temps.append(seasonal+trend+noise)

    df = pd.DataFrame({"date":date_array, "temperature":temps})

    # extract year, month, days
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day

    return df