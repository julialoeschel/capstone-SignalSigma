# ---
# description: Streamlit app to visualize macroeconomic composite signals from FRED.
# ---

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# --- Load your preprocessed macro data ---
@st.cache_data
def load_fred_composites():
    return pd.read_csv(
        "../data/fed/fred_macro_composite_3features.csv",
        parse_dates=True,
        # XXX: Value missing!
        # index_col=
    )


# Load the data
df = load_fred_composites()

# --- Streamlit UI ---
st.title("📈 FRED Composite Macro Feature Visualizer")
st.write(
    """
Visualize 3 interpretable macroeconomic composite signals:
- **Inflation & Monetary Pressure**
- **Labor & Economic Activity**
- **Consumer Spending & Sentiment**
"""
)

# Show full DataFrame
if st.checkbox("Show Raw Composite Data"):
    st.dataframe(df.tail(10))

# Select one or more features to visualize
selected_cols = st.multiselect(
    "Select features to plot",
    options=df.columns.tolist(),
    default=df.columns.tolist(),
)

# Date range filter
min_date, max_date = df.index.min(), df.index.max()
date_range = st.slider(
    "Select Date Range",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
)
filtered_df = df.loc[date_range[0] : date_range[1]]

# Plot selected features
if selected_cols:
    st.subheader("📊 Time Series Plot")
    fig, ax = plt.subplots(figsize=(12, 5))
    filtered_df[selected_cols].plot(ax=ax, linewidth=2)
    plt.title("Composite Macro Feature Trends")
    plt.xlabel("Date")
    plt.ylabel("Composite Value (Standardized)")
    plt.grid(True)
    st.pyplot(fig)
else:
    st.warning("Please select at least one feature.")

# Optional: correlation heatmap
if st.checkbox("Show Correlation Heatmap"):
    st.subheader("🔗 Correlation Between Macro Features")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# Optional: download
st.download_button(
    label="📥 Download Selected Data as CSV",
    data=filtered_df[selected_cols].to_csv().encode(),
    file_name="fred_macro_composites_filtered.csv",
)
