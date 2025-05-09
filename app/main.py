import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model
from generic_stock_prediction import (
    load_stock_data,
    prepare_data,
    build_model,
    predict_future
)

# Create models directory
MODEL_DIR = 'models'
if not os.path.isdir(MODEL_DIR):
    os.makedirs(MODEL_DIR)

st.set_page_config(page_title="Stock Price Predictor", layout="wide")
st.title("📈 Stock Price Prediction App")

# Sidebar inputs
st.sidebar.header("Model Parameters")
# Sidebar: select from Magnificent Seven stocks or custom
MAG7 = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA']
ticker_option = st.sidebar.selectbox("Ticker symbol", options=MAG7 + ['Other'])
if ticker_option == 'Other':
    ticker = st.sidebar.text_input("Enter custom ticker", value="AAPL").upper()
else:
    ticker = ticker_option

start_date = st.sidebar.date_input("Start date", value=pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End date", value=None)
seq_len = st.sidebar.number_input("Sequence length", min_value=1, max_value=200, value=60)
epochs = st.sidebar.number_input("Epochs", min_value=1, max_value=100, value=10)
batch_size = st.sidebar.number_input("Batch size", min_value=1, max_value=256, value=32)
predict_days = st.sidebar.number_input("Days to predict", min_value=1, max_value=30, value=1)

generate = st.sidebar.button("🔮 Generate Forecast")

st.subheader(f"Forecast for {ticker}")

if generate:
    # Load and prepare data
    try:
        df = load_stock_data(
            ticker,
            start=start_date.isoformat(),
            end=end_date.isoformat() if end_date else None
        )
        X, y, scaler = prepare_data(df, seq_len=seq_len)
    except Exception as e:
        st.error(f"Data loading error: {e}")
        st.stop()

    # Model file paths
    model_file = os.path.join(MODEL_DIR, f"{ticker}_model.h5")
    scaler_file = os.path.join(MODEL_DIR, f"{ticker}_scaler.pkl")

    # Train or load model
    if not os.path.isfile(model_file) or not os.path.isfile(scaler_file):
        with st.spinner("Training model (this may take a while)..."):
            model = build_model((X.shape[1], 1))
            model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
            model.save(model_file)
            joblib.dump(scaler, scaler_file)
        st.success("Training complete and model saved.")
    else:
        with st.spinner("Loading saved model..."):
            model = load_model(model_file)
            scaler = joblib.load(scaler_file)
        st.success("Model loaded.")

    # Evaluate performance on a test split
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import numpy as np
    split_idx = int(len(X) * 0.8)
    X_test = X[split_idx:]
    y_test = y[split_idx:]
    if len(X_test) > 0:
        y_pred_scaled = model.predict(X_test, verbose=0)
        y_pred = scaler.inverse_transform(y_pred_scaled)
        y_true = scaler.inverse_transform(y_test.reshape(-1, 1))
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        st.subheader("Model Performance on Test Set")
        col1, col2, col3 = st.columns(3)
        col1.metric("RMSE", f"{rmse:.2f}")
        col2.metric("MAE", f"{mae:.2f}")
        col3.metric("MSE", f"{mse:.2f}")
    else:
        st.warning("Not enough data for performance evaluation.")

    # Generate predictions
    last_seq = X[-1].flatten()
    future_preds = predict_future(model, last_seq, scaler, n_days=predict_days)
    next_start = df.index[-1] + pd.Timedelta(days=1)
    future_dates = pd.bdate_range(start=next_start, periods=predict_days)
    result = pd.DataFrame({'Date': future_dates, 'Predicted_Close': future_preds}).set_index('Date')

    # Display results
    st.dataframe(result)
    # Download forecast
    csv = result.to_csv().encode('utf-8')
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name=f"{ticker}_forecast.csv",
        mime="text/csv"
    )

    # Interactive Plot with Plotly: Forecast
    import plotly.express as px
    df_forecast = result.reset_index()
    fig1 = px.line(
        df_forecast,
        x='Date',
        y='Predicted_Close',
        markers=True,
        title=f"Next {predict_days} Day Forecast for {ticker}",
        hover_data={'Date':True, 'Predicted_Close':':.2f'}
    )
    fig1.update_traces(hovertemplate='Date: %{x}<br>Predicted Close: %{y:.2f}')
    fig1.update_layout(xaxis_title='Date', yaxis_title='Close Price')
    st.plotly_chart(fig1, use_container_width=True)

    # Interactive Overlay with Plotly: Historical vs Forecast
    df_hist = df[['Close']].reset_index().rename(columns={'Close':'Price'})
    df_hist['Type'] = 'Historical'
    df_fc = df_forecast.rename(columns={'Predicted_Close':'Price'})
    df_fc['Type'] = 'Forecast'
    df_combined = pd.concat([df_hist, df_fc], ignore_index=True)
    fig2 = px.line(
        df_combined,
        x='Date',
        y='Price',
        color='Type',
        markers=True,
        color_discrete_map={'Historical':'blue', 'Forecast':'red'},
        title=f"Historical and Forecast Close Price for {ticker}",
        hover_data={'Date':True, 'Price':':.2f', 'Type':False}
    )
    fig2.update_traces(hovertemplate='Date: %{x}<br>Price: %{y:.2f}')
    fig2.update_layout(xaxis_title='Date', yaxis_title='Close Price')
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("Adjust parameters in the sidebar and click 'Generate Forecast'.")
