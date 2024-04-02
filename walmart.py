import streamlit as st
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

def load_data():
    data = pd.read_csv("Walmart_new.csv")
    return data

def arima_forecast(data):
    years = list(data['Year']) + list(range(data['Year'].iloc[-1] + 1, data['Year'].iloc[-1] + 6))
    model = ARIMA(data['Net_sales'], order=(5, 1, 0))
    model_fit = model.fit()
    forecast_arima = model_fit.forecast(steps=5)
    forecast_df = pd.DataFrame({'Year': years[-len(forecast_arima):], 'Forecast': forecast_arima})
    return forecast_df, forecast_arima

def lstm_forecast(data, future_years):
    n_features = 1
    series = np.array(data['Net_sales']).reshape((len(data), n_features))

    X, y = list(), list()
    n_steps = 3
    for i in range(len(series)):
        end_ix = i + n_steps
        if end_ix > len(series) - 1:
            break
        seq_x, seq_y = series[i:end_ix], series[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    X = np.array(X)
    y = np.array(y)

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=200, verbose=0)

    forecasts_lstm = []
    x_input = X[-1:]
    for _ in range(len(future_years)):
        yhat = model.predict(x_input, verbose=0)
        forecasts_lstm.append(yhat[0][0])
        x_input = np.append(x_input[0][1:], yhat).reshape((1, n_steps, n_features))

    forecasts_lstm_df = pd.DataFrame({'Year': future_years, 'Forecast': forecasts_lstm})
    return forecasts_lstm_df, forecasts_lstm

def main():
    st.title("Walmart Sales Forecasting")
    data = load_data()
    st.write("Raw data:")
    st.write(data.head())

    forecast_df, forecast_arima = arima_forecast(data)
    st.write("ARIMA forecast:")
    st.write(forecast_df)

    future_years = list(range(data['Year'].iloc[-1] + 1, data['Year'].iloc[-1] + 11))
    forecasts_lstm_df, forecasts_lstm = lstm_forecast(data, future_years)
    st.write("LSTM forecast:")
    st.write(forecasts_lstm_df)

    arima_mse = mean_squared_error(data['Net_sales'].iloc[-len(forecast_arima):], forecast_arima)
    lstm_mse = mean_squared_error(data['Net_sales'].iloc[-len(forecasts_lstm):], forecasts_lstm[:len(data) - len(forecasts_lstm_df)])

    arima_mape = mean_absolute_percentage_error(data['Net_sales'].iloc[-len(forecast_arima):], forecast_arima)
    lstm_mape = mean_absolute_percentage_error(data['Net_sales'].iloc[-len(forecasts_lstm):], forecasts_lstm[:len(data) - len(forecasts_lstm_df)])

    st.write(f"ARIMA MSE: {arima_mse:.2f}")
    st.write(f"LSTM MSE: {lstm_mse:.2f}")
    st.write(f"ARIMA MAPE: {arima_mape:.2%} ({100 - arima_mape * 100:.2f}% accuracy)")
    st.write(f"LSTM MAPE: {lstm_mape:.2%} ({100 - lstm_mape * 100:.2f}% accuracy)")

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.plot(data['Year'], data['Net_sales'], label='Actual')
    ax.plot(forecast_df['Year'], forecast_df['Forecast'], label='ARIMA Forecast')
    ax.plot(forecasts_lstm_df['Year'], forecasts_lstm_df['Forecast'], label='LSTM Forecast')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.legend()
    st.pyplot(fig)

if __name__ == "__main__":
    main()
