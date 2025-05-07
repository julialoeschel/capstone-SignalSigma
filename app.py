import streamlit as st
import pandas as pd
import plotly.express as px


df = pd.read_csv("data/processed_combined_data.csv", parse_dates=["date"])

stock_columns = [col for col in df.columns if col.startswith("close_")]

stock_names = [col.replace("close_", "") for col in stock_columns[:10]]

st.title("📈 Aktien-Dashboard")

selected_stock = st.selectbox("Wähle eine Aktie:", stock_names)

column_name = f"close_{selected_stock}"

fig = px.line(df, x="date", y=column_name, title=f"Verlauf von {selected_stock} (Close)")
st.plotly_chart(fig)

st.metric("Letzter Preis", f"${df[column_name].iloc[-1]:.2f}")