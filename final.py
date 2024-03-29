import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import load_model
import os

# Load data from CSV file
@st.cache_data
def load_data(dataset):
    df = pd.read_csv(dataset, parse_dates=True, usecols=["Date", "Weekly_Sales"])
    df.dropna()
    df['Date'] = pd.to_datetime(df['Date'],format='mixed', dayfirst=True)
    return df

# LSTM model
def run_lstm(df):
    train_size = int(len(df) * 0.8)
    train = df[:train_size]
    test = df[train_size:]

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train['Weekly_Sales'].values.reshape(-1, 1))
    test_scaled = scaler.transform(test['Weekly_Sales'].values.reshape(-1, 1))

    time_steps = 5

    X_train, y_train = prepare_data(train_scaled, time_steps)
    X_test, y_test = prepare_data(test_scaled, time_steps)

    model_path = 'lstm_model.h5'
    
    if os.path.exists(model_path):
        # Load the saved model if it exists
        model = load_model(model_path)
    else:
        # Train the model and save it
        model = Sequential()
        model.add(LSTM(units=50, activation='relu', input_shape=(time_steps, 1)))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)
        model.save(model_path)

    test_predictions = model.predict(X_test)
    test_predictions = scaler.inverse_transform(test_predictions)

    test_rmse = np.sqrt(mean_squared_error(test['Weekly_Sales'][time_steps:], test_predictions[:, 0]))
    test_mape = np.mean(np.abs((test['Weekly_Sales'][time_steps:].values - test_predictions[:, 0]) / test['Weekly_Sales'][time_steps:].values)) * 100

    return test.index[time_steps:], test['Weekly_Sales'][time_steps:], test_predictions[:, 0], test_mape

def prepare_data(data, time_steps=1):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i+time_steps, 0])
        y.append(data[i+time_steps, 0])
    return np.array(X), np.array(y)

def main():
    st.title("Sales Forecasting App")

    dataset = st.sidebar.selectbox("Select Dataset", ["Walmart.csv", "data1.csv", "data2.csv", "data3.csv"])
    df = load_data(dataset)

    model_type = st.sidebar.selectbox("Select Model", ["LSTM"])

    if model_type == "LSTM":
        test_dates, test_actual, test_predictions, test_mape = run_lstm(df)

        st.subheader("LSTM Model Results")
        st.write(f"Test MAPE: {test_mape:.2f}%")
        st.write(f"Average Accuracy: {100 - test_mape:.2f}%")

        fig, ax = plt.subplots()
        ax.plot(test_dates, test_actual, label='Actual')
        ax.plot(test_dates, test_predictions, label='LSTM Predicted')
        ax.legend()
        st.pyplot(fig)

if __name__ == "__main__":
    main()
