import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import datetime
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(
    page_title="Stock Price Forecast Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ===========================================
# MODEL FUNCTIONS (Define these first!)
# ===========================================

def add_features(df, ticker='M'):
    close_col = f'Close_{ticker}'
    volume_col = f'Volume_{ticker}'

    ticker_df = pd.DataFrame({
        'Close': df[close_col],
        'Volume': df[volume_col],
    })

    # Calculate moving averages
    ticker_df['MA_5'] = ticker_df['Close'].rolling(window=5).mean()
    ticker_df['MA_10'] = ticker_df['Close'].rolling(window=10).mean()
    ticker_df['MA_100'] = ticker_df['Close'].rolling(window=100).mean()

    ticker_df['Price_Delta'] = ticker_df['Close'].diff()
    ticker_df['Pct_Change'] = ticker_df['Close'].pct_change() * 100
    ticker_df.dropna(inplace=True)
    return ticker_df


def preprocess_data(ticker_df):
    ticker_df['Target'] = ticker_df['Close'].shift(-1)
    ticker_df.dropna(inplace=True)

    features = ['Close', 'Volume', 'MA_5', 'MA_10', 'MA_100', 'Price_Delta', 'Pct_Change']
    X = ticker_df[features]
    y = ticker_df['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, X_test.index


def preprocess_data_lstm(ticker_df, time_steps=10):
    ticker_df['Target'] = ticker_df['Close'].shift(-1)
    ticker_df.dropna(inplace=True)

    features = ['Close', 'Volume', 'MA_5', 'MA_10', 'MA_100', 'Price_Delta', 'Pct_Change']
    X = ticker_df[features].values
    y = ticker_df['Target'].values

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - time_steps):
        X_seq.append(X_scaled[i:(i + time_steps)])
        y_seq.append(y[i + time_steps])
    X_seq, y_seq = np.array(X_seq), np.array(y_seq)

    train_size = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:train_size], X_seq[train_size:]
    y_train, y_test = y_seq[:train_size], y_seq[train_size:]
    test_dates = ticker_df.index[train_size + time_steps:]

    # Convert to pandas Series
    y_train = pd.Series(y_train.flatten())
    y_test = pd.Series(y_test.flatten())

    return X_train, X_test, y_train, y_test, test_dates


def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    model = XGBRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_lstm(X_train, y_train, time_steps=10):
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(time_steps, X_train.shape[2]), return_sequences=False),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    return model


# ===========================================
# DATA LOADING AND CLEANING
# ===========================================

@st.cache_data
def load_and_clean_data():
    try:
        # Load the CSV file
        df = pd.read_csv('MVR.csv', parse_dates=['Date'])

        # Sort by date and set as index
        df = df.sort_values('Date')
        df.set_index('Date', inplace=True)

        # Drop missing values
        df.dropna(inplace=True)

        # Columns to clean
        price_columns = [
            'Open_M', 'High_M', 'Low_M', 'Close_M', 'Adj Close_M',
            'Open_V', 'High_V', 'Low_V', 'Close_V', 'Adj Close_V'
        ]

        # Clean price columns: remove $, convert to float
        for col in price_columns:
            df[col] = df[col].replace('[\$,]', '', regex=True).astype(float)

        # Clean volume columns: remove ' M', convert to float
        df['Volume_M'] = df['Volume_M'].replace('[\sM]', '', regex=True).astype(float)
        df['Volume_V'] = df['Volume_V'].replace('[\sM]', '', regex=True).astype(float)

        # Remove outliers using z-score on Close_M and Close_V
        z_close_m = np.abs(stats.zscore(df['Close_M']))
        z_close_v = np.abs(stats.zscore(df['Close_V']))
        df = df[(z_close_m < 3) & (z_close_v < 3)]

        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Create sample data for demonstration
        dates = pd.date_range('2008-01-01', '2024-12-31', freq='D')
        df = pd.DataFrame({
            'Close_M': np.random.normal(200, 50, len(dates)).cumsum(),
            'Close_V': np.random.normal(150, 40, len(dates)).cumsum(),
            'Volume_M': np.random.normal(1000000, 200000, len(dates)),
            'Volume_V': np.random.normal(800000, 150000, len(dates))
        }, index=dates)
        st.warning("Using sample data for demonstration")
        return df


# Load data
df = load_and_clean_data()

# ===========================================
# SIDEBAR CONTROLS
# ===========================================

st.sidebar.title("Dashboard Controls")

# Stock selection
ticker = st.sidebar.selectbox(
    "Select Stock:",
    ["Mastercard (M)", "Visa (V)"],
    index=0
)
ticker_symbol = 'M' if 'Mastercard' in ticker else 'V'

# Model selection
model_choice = st.sidebar.selectbox(
    "Select Model:",
    ["Random Forest", "Linear Regression", "XGBoost", "LSTM"],
    index=1
)


# ===========================================
# MODEL TRAINING FUNCTION (Cached)
# ===========================================

@st.cache_resource
def train_models(_df, ticker):
    """Train all models and cache results"""
    try:
        ticker_df = add_features(_df, ticker)

        # Train traditional models
        X_train, X_test, y_train, y_test, test_dates = preprocess_data(ticker_df)

        models = {
            'Random Forest': train_random_forest(X_train, y_train),
            'Linear Regression': train_linear_regression(X_train, y_train),
            'XGBoost': train_xgboost(X_train, y_train)
        }

        # Train LSTM
        X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm, test_dates_lstm = preprocess_data_lstm(ticker_df)
        models['LSTM'] = train_lstm(X_train_lstm, y_train_lstm)

        return models, {
            'X_test': X_test,
            'y_test': y_test,
            'test_dates': test_dates,
            'X_test_lstm': X_test_lstm,
            'y_test_lstm': y_test_lstm,
            'test_dates_lstm': test_dates_lstm
        }
    except Exception as e:
        st.error(f"Error training models: {e}")
        return {}, {}


# ===========================================
# MAIN DASHBOARD
# ===========================================

st.title("📈 Stock Price Forecast Dashboard")
st.markdown("### Interactive analysis of Mastercard and Visa stock predictions")

# Display basic info
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Data Points", len(df))
with col2:
    st.metric("Start Date", df.index.min().strftime('%Y-%m-%d'))
with col3:
    st.metric("End Date", df.index.max().strftime('%Y-%m-%d'))

# ===========================================
# PRICE CHART
# ===========================================

st.subheader(f"{ticker} Price History")
fig_price = go.Figure()
fig_price.add_trace(go.Scatter(
    x=df.index,
    y=df[f'Close_{ticker_symbol}'],
    mode='lines',
    name=f'{ticker_symbol} Close Price',
    line=dict(color='#1f77b4' if ticker_symbol == 'M' else '#2ca02c')
))
fig_price.update_layout(
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    hovermode="x unified"
)
st.plotly_chart(fig_price, use_container_width=True)


# ===========================================
# MODEL TRAINING FUNCTION (Cached)
# ===========================================

@st.cache_resource
def train_models(_df, ticker):
    """Train all models and cache results"""
    try:
        ticker_df = add_features(_df, ticker)

        # Train traditional models
        X_train, X_test, y_train, y_test, test_dates = preprocess_data(ticker_df)

        models = {
            'Random Forest': train_random_forest(X_train, y_train),
            'Linear Regression': train_linear_regression(X_train, y_train),
            'XGBoost': train_xgboost(X_train, y_train)
        }

        # Train LSTM
        X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm, test_dates_lstm = preprocess_data_lstm(ticker_df)
        models['LSTM'] = train_lstm(X_train_lstm, y_train_lstm)

        # Ensure all arrays have the same length for traditional models
        min_length_trad = min(len(X_test), len(y_test), len(test_dates))

        # Ensure all arrays have the same length for LSTM
        min_length_lstm = min(len(X_test_lstm), len(y_test_lstm), len(test_dates_lstm))

        return models, {
            'X_test': X_test[:min_length_trad],
            'y_test': y_test.iloc[:min_length_trad],
            'test_dates': test_dates[:min_length_trad],
            'X_test_lstm': X_test_lstm[:min_length_lstm],
            'y_test_lstm': y_test_lstm.iloc[:min_length_lstm],
            'test_dates_lstm': test_dates_lstm[:min_length_lstm]
        }
    except Exception as e:
        st.error(f"Error training models: {e}")
        return {}, {}


# ===========================================
# MODEL TRAINING AND PREDICTION
# ===========================================

st.subheader(f"Model Predictions - {model_choice}")

# Train models (cached)
models, test_data = train_models(df, ticker_symbol)

if models and test_data:
    # Make predictions
    if model_choice == "LSTM":
        X_test = test_data['X_test_lstm']
        y_test = test_data['y_test_lstm']
        test_dates = test_data['test_dates_lstm']
        is_lstm = True
    else:
        X_test = test_data['X_test']
        y_test = test_data['y_test']
        test_dates = test_data['test_dates']
        is_lstm = False

    model = models[model_choice]
    y_pred = model.predict(X_test)

    # Convert to pandas Series and ensure proper length
    if isinstance(y_test, np.ndarray):
        y_test = pd.Series(y_test.flatten())
    if isinstance(y_pred, np.ndarray):
        y_pred = pd.Series(y_pred.flatten())

    # Ensure all arrays have the same length
    min_length = min(len(test_dates), len(y_test), len(y_pred))
    test_dates = test_dates[:min_length]
    y_test = y_test.iloc[:min_length] if hasattr(y_test, 'iloc') else y_test[:min_length]
    y_pred = y_pred[:min_length]

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("RMSE", f"{rmse:.4f}")
    col2.metric("MAE", f"{mae:.4f}")
    col3.metric("R² Score", f"{r2:.4f}")
    col4.metric("Model", model_choice)

    # ===========================================
    # INTERACTIVE PREDICTION CHART
    # ===========================================

    fig_pred = go.Figure()

    # Actual prices
    fig_pred.add_trace(go.Scatter(
        x=test_dates,
        y=y_test,
        mode='lines',
        name='Actual Prices',
        line=dict(color='#1f77b4' if ticker_symbol == 'M' else '#2ca02c', width=3)
    ))

    # Predicted prices
    fig_pred.add_trace(go.Scatter(
        x=test_dates,
        y=y_pred,
        mode='lines',
        name='Predicted Prices',
        line=dict(color='#d62728', width=2, dash='dash')
    ))

    fig_pred.update_layout(
        title=f"{ticker} - {model_choice} Predictions",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    st.plotly_chart(fig_pred, use_container_width=True)

    # ===========================================
    # RESIDUAL ANALYSIS
    # ===========================================

    st.subheader("Residual Analysis")

    residuals = y_test - y_pred

    fig_residuals = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Residual Distribution', 'Residuals vs Predicted')
    )

    # Histogram of residuals
    fig_residuals.add_trace(
        go.Histogram(x=residuals, nbinsx=50, name='Residuals'),
        row=1, col=1
    )

    # Residuals vs Predicted
    fig_residuals.add_trace(
        go.Scatter(x=y_pred, y=residuals, mode='markers', name='Residuals'),
        row=1, col=2
    )

    fig_residuals.update_layout(
        title_text="Residual Analysis",
        showlegend=False
    )
    fig_residuals.update_xaxes(title_text="Residual Value", row=1, col=1)
    fig_residuals.update_yaxes(title_text="Frequency", row=1, col=1)
    fig_residuals.update_xaxes(title_text="Predicted Values", row=1, col=2)
    fig_residuals.update_yaxes(title_text="Residuals", row=1, col=2)

    st.plotly_chart(fig_residuals, use_container_width=True)

    # ===========================================
    # DOWNLOAD PREDICTIONS
    # ===========================================

    st.subheader("Download Predictions")

    # Create prediction dataframe - ensure all arrays have same length
    min_length = min(len(test_dates), len(y_test), len(y_pred))

    predictions_df = pd.DataFrame({
        'Date': test_dates[:min_length],
        'Actual_Price': y_test.values[:min_length] if hasattr(y_test, 'values') else y_test[:min_length],
        'Predicted_Price': y_pred.values[:min_length] if hasattr(y_pred, 'values') else y_pred[:min_length],
        'Residual': (y_test.values[:min_length] - y_pred.values[:min_length]) if hasattr(y_test, 'values') else (
                    y_test[:min_length] - y_pred[:min_length])
    })

    # Display sample data
    st.dataframe(predictions_df.head(), use_container_width=True)

    # Download button
    csv = predictions_df.to_csv(index=False)
    st.download_button(
        label="Download Predictions as CSV",
        data=csv,
        file_name=f"{ticker_symbol}_{model_choice}_predictions.csv",
        mime="text/csv"
    )
else:
    st.warning("Could not train models. Please check your data.")
# ===========================================
# MOVING AVERAGES SECTION
# ===========================================

st.subheader("Moving Averages Analysis")

# Calculate moving averages
df_ma = df.copy()
df_ma[f'{ticker_symbol}_MA_5'] = df_ma[f'Close_{ticker_symbol}'].rolling(window=5).mean()
df_ma[f'{ticker_symbol}_MA_10'] = df_ma[f'Close_{ticker_symbol}'].rolling(window=10).mean()
df_ma[f'{ticker_symbol}_MA_20'] = df_ma[f'Close_{ticker_symbol}'].rolling(window=20).mean()

fig_ma = go.Figure()
fig_ma.add_trace(go.Scatter(
    x=df_ma.index, y=df_ma[f'Close_{ticker_symbol}'],
    name='Close Price', line=dict(color='black', width=2)
))
fig_ma.add_trace(go.Scatter(
    x=df_ma.index, y=df_ma[f'{ticker_symbol}_MA_5'],
    name='5-Day MA', line=dict(color='blue', width=1)
))
fig_ma.add_trace(go.Scatter(
    x=df_ma.index, y=df_ma[f'{ticker_symbol}_MA_10'],
    name='10-Day MA', line=dict(color='green', width=1)
))
fig_ma.add_trace(go.Scatter(
    x=df_ma.index, y=df_ma[f'{ticker_symbol}_MA_20'],
    name='20-Day MA', line=dict(color='red', width=1)
))

fig_ma.update_layout(
    title=f"{ticker} Moving Averages",
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    hovermode="x unified"
)

st.plotly_chart(fig_ma, use_container_width=True)

# ===========================================
# MODEL COMPARISON
# ===========================================

st.subheader("Model Performance Comparison")

# Define metrics for all models
metrics_data = {
    'Random Forest': {'M': {'RMSE': 119.8063, 'MAE': 101.1172, 'R2': -2.4841},
                      'V': {'RMSE': 71.2955, 'MAE': 61.9712, 'R2': -3.087}},
    'Linear Regression': {'M': {'RMSE': 5.9330, 'MAE': 2.9700, 'R2': 0.9915},
                          'V': {'RMSE': 3.1989, 'MAE': 2.0385, 'R2': 0.9918}},
    'XGBoost': {'M': {'RMSE': 118.9578, 'MAE': 100.1136, 'R2': -2.4349},
                'V': {'RMSE': 73.2595, 'MAE': 64.2726, 'R2': -3.3153}},
    'LSTM': {'M': {'RMSE': 167.5146, 'MAE': 155.1311, 'R2': -6.0236},
             'V': {'RMSE': 115.8975, 'MAE': 110.5971, 'R2': -10.1884}}
}

# Create comparison chart
models_list = list(metrics_data.keys())
metrics = ['RMSE', 'MAE', 'R2']

fig_comparison = make_subplots(
    rows=1, cols=3,
    subplot_titles=('RMSE Comparison', 'MAE Comparison', 'R² Comparison')
)

for i, metric in enumerate(metrics):
    values = [metrics_data[model][ticker_symbol][metric] for model in models_list]
    fig_comparison.add_trace(
        go.Bar(x=models_list, y=values, name=metric),
        row=1, col=i + 1
    )
    fig_comparison.update_yaxes(title_text=metric, row=1, col=i + 1)

fig_comparison.update_layout(
    title_text=f"Model Performance Comparison for {ticker}",
    showlegend=False,
    height=400
)

st.plotly_chart(fig_comparison, use_container_width=True)

# ===========================================
# FOOTER
# ===========================================

st.markdown("---")
st.markdown("""
**Note:** This dashboard provides interactive analysis of stock price predictions. 
Select different models and stocks from the sidebar to compare performance.
""")