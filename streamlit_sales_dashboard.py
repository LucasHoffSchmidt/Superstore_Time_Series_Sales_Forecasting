# Importing packages
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pmdarima as pm
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error
import joblib

# App title
st.title("Superstore Sales Data Interactive Dashboard")

# Loading and caching data
@st.cache_data
def load_data():
    sales_df = joblib.load("data/sales_df.pkl")
    sales_monthly = joblib.load("data/sales_monthly.pkl")
    X_monthly = joblib.load("data/X_monthly.pkl")

    return sales_df, sales_monthly, X_monthly

sales_df, sales_monthly, X_monthly = load_data()

@st.cache_resource
def load_model():
    return joblib.load("model/arimax_model.pkl")

model_pretrained = load_model()

# --- SECTION 1: Filtered Segment Sales ---    
# Making sidebar filters
st.sidebar.header("Filters")
selected_region = st.sidebar.multiselect("Region", sales_df["Region"].unique(), default=sales_df["Region"].unique())
selected_category = st.sidebar.multiselect("Category", sales_df["Category"].unique(), default=sales_df["Category"].unique())
selected_year = st.sidebar.multiselect("Year", sales_df["Year"].unique(), default=sales_df["Year"].unique())
selected_month = st.sidebar.multiselect("Month", sales_df["Month"].unique(), default=sales_df["Month"].unique())

# Filtering sales dataset
filtered_sales = sales_df[
    (sales_df["Region"].isin(selected_region)) &
    (sales_df["Category"].isin(selected_category)) &
    (sales_df["Year"].isin(selected_year)) &
    (sales_df["Month"].isin(selected_month))
]

# Showing filtered dataset
st.subheader("Filtered Sales")
st.write(filtered_sales.head())

# Defining segment order
segment_order = ["Consumer", "Corporate", "Home Office"]
segment_sales = (
    filtered_sales
    .groupby("Segment")["Sales"]
    .sum()
    .reindex(segment_order)
    .reset_index()
)

# Plotting filtered segment sales
fig = plt.figure(figsize=(10, 6))
sns.barplot(data=segment_sales, x="Segment", y="Sales")

plt.title("Total sales by segment")
plt.xlabel("Segment")
plt.ylabel("Sales")
st.pyplot(fig)

# --- SECTION 2: ARIMAX Forecast vs Actual Sales  --- 
st.subheader("Sales Forecasting with the ARIMAX Model")

start_month = (sales_monthly.index[-1] - pd.DateOffset(months=12)).date()
max_month = sales_monthly.index[-1].date()

# Converting to time period to ensure accurate indexing
start_month_ts = pd.Timestamp(start_month) + pd.offsets.MonthEnd(0)

# Making a slider for the forecast horizon
forecast_horizon = st.slider(
    "Select forecasting period (1-12 months):", 
    min_value = 1, 
    max_value = 12,
    value = 1, 
    step = 1
)

# Forecasting functions
# Creating evaluation function
def evaluate_model(y_true, y_pred, forecast_horizon):
    mae = mean_absolute_error(y_true, y_pred) # Absolute forecasting errors
    rmse = root_mean_squared_error(y_true, y_pred) # Penalizes larger forecasting errors
    mape = mean_absolute_percentage_error(y_true, y_pred) # Relative forecasting errors
    forecast_bias = np.mean(y_pred - y_true) # Indicates tendency to overpredict or underpredict

    evaluations = (
        f"ARIMAX model results:\n"
        f"Forecast horizon: {forecast_horizon} months\n"
        f"mae: {mae}\n"
        f"mape: {mape}\n"
        f"rmse:{rmse}\n"
        f"forecast_bias: {forecast_bias}"
    )

    return evaluations

# Creating forecast function
def forecast_arimax(sales_monthly, X_monthly, start_month, forecast_horizon=1):    
    # Getting the forecast boundaries
    window_start = sales_monthly.index.get_loc(start_month)
    window_end = window_start + forecast_horizon - 1
    
    # Getting the true sales values for the forecasting horizon
    y_true = sales_monthly["Sales"].iloc[window_start:window_end + 1]
    
    # Splitting exogenous variables for the forecast horizon
    X_test = X_monthly.iloc[window_start:window_end + 1]

    # Load pretrained ARIMAX model
    arimax_model = model_pretrained
    
    # Forecasting the logged sales values for the forecasting horizon
    arimax_log_forecast = arimax_model.predict(n_periods=forecast_horizon, X=X_test)
    
    # Inverse transformation to get sales values in the original scale
    arimax_forecast = np.expm1(arimax_log_forecast)

    return arimax_forecast, y_true

if st.button("Run Forecast"):    
    # Making a ARIMAX forecast
    arimax_forecast, y_true = forecast_arimax(
        sales_monthly, X_monthly, start_month=start_month_ts, forecast_horizon = forecast_horizon
    )
        
    # Evaluating the model
    arimax_evaluation = evaluate_model(y_true, arimax_forecast, forecast_horizon)
    st.text(arimax_evaluation)

    # Plotting actual sales data for the last 24 months
    figure = plt.figure(figsize=(12,6))
    plt.plot(arimax_forecast.index.strftime('%Y-%m'), y_true, marker="o", 
             label=f"Actual sales", color="blue")
        
    # Plotting ARIMAX forecast
    plt.plot(arimax_forecast.index.strftime('%Y-%m'), arimax_forecast, marker="o", 
             label=f"ARIMAX forecast with {forecast_horizon} months horizon ", color="red", 
             linestyle="dashed")
        
    # Formatting plot
    plt.title(f"ARIMAX Forecast vs Actual Sales")
    plt.xlabel("Date")
    plt.xticks(rotation=45)
    plt.ylabel("Sales")
    plt.legend()
    plt.grid(True)
    st.pyplot(figure)
    
    # --- SECTION 3: Residuals of Forecasted Sales  ---        
    st.subheader("Residuals of Forecasted Sales")

    # Calculating residuals between true sales values and forecasted sales values
    residuals = arimax_forecast.values - y_true
    
    # Plotting residuals
    fig = plt.figure(figsize=(12, 6))
    plt.plot(arimax_forecast.index.strftime('%Y-%m'), residuals, label="Residuals", color="green")
    plt.axhline(0, color="black", linestyle="--")
    plt.title("Residuals of ARIMAX Forecast")
    plt.xlabel("Month")
    plt.xticks(rotation=45)
    plt.ylabel("Residuals")
    plt.legend()
    plt.grid(True)
    st.pyplot(fig)

    # --- SECTION 4: Percentage Errors of Forecasted Sales  ---    
    st.subheader("Percentage Errors of Forecasted Sales")
    
    # Calculating percentage errors and MAPE for reference
    error_percentages = np.abs(residuals / y_true) * 100
    mape = np.mean(np.abs(error_percentages))
    
    # Plotting percentage errors of forecasted values
    fig = plt.figure(figsize=(10, 6))
    sns.barplot(x=arimax_forecast.index.strftime('%Y-%m'), y=error_percentages, color="purple")
    plt.axhline(y=mape, color="red", linestyle="--", label=f"Mean Absolute Percentage Error: {mape:.2f}%")
    plt.title("Percentage Error Of Forecasted Sales")
    plt.xlabel("Month")
    plt.xticks(rotation=45)
    plt.ylabel("Percentage Error")
    plt.legend()
    plt.grid(True)
    st.pyplot(fig)