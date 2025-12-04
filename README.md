# Superstore Time Series Sales Forecasting
Data science project that seeks to make a sales forecasting model to predict future sales for a global superstore to assist in optimizing pricing and marketing strategies. 

## Quick Links
- Streamlit app for interactive data and model predictions analysis: [Streamlit App](https://superstore-time-series-sales-forecasting.streamlit.app/)
- Jupyter Notebook of the development of the sales forecasting model: [Jupyter Notebook](sales_forecast.ipynb)
- Superstore sales dataset: [Superstore Sales Dataset](sales-forecasting-dataset/train.csv)
- Other projects I have made: [Portfolio Website](https://lucashoffschmidt.github.io/)

## Technologies Used
**Tools and Platforms**
- Development: Jupyterlab
- Deployment: Streamlit Community Cloud

**Libraries**
- Dataset Handling: `opendataset`
- Data Analysis: `pandas`, `numpy`
- Visualization: `matplotlib`, `seaborn`, `folium`
- Machine Learning: `scikit-learn`
- Statistical Modeling: `statsmodels`, `pmdarima`
- File System Operations: `os`
- Date and Time Handling: `datetime`
- Storage: `joblib`
- Deployment: `streamlit`

## Process
**Data Collection**
- Acquired the superstore sales dataset from kaggle, renamed the dataset folder and converted the dataset to a pandas DataFrame.
<img src="images/dataframe.jpg" alt="DataFrame" width="800">
- Defined the attributes of the dataset. <br><br>

**Exploratory Data Analysis**
- Checked the dataframe for null values and incorrect data types and converted the date features Order Date and Ship Date to datetime format.
- Investigated the statistical distribution of attributes and created a histogram and boxplot of sales to check for skewness and outliers.
<img src="images/histogram.jpg" alt="Histogram of sales" width="700">
<img src="images/boxplot.jpg" alt="Boxplot of sales" width="700">
- Created new datetime features such as month and year and used them in lineplots to see how total sales fluctuate over time.
<img src="images/months_elapsed.jpg" alt="Total sales by months elapsed" width="700">
- Visualized how categorical features such as segment and region relate to total sales using barplots.
<img src="images/total_sales_region.jpg" alt="Total sales by segment and region" width="700">
<img src="images/total_sales_category.jpg" alt="Total sales by category and subcategory" width="700">
- Created an interactive map of total sales by state.
<img src="images/states_map.jpg" alt="Total sales by state" width="700"><br>

**Data Preprocessing**
- Checked the dataframe for duplicates, invalid names and extra spaces.
- Dropped any feature that is not suitable for modeling, such as high cardinality features and features that are unknown at the time of forecasting. <br><br>

**Model Training and Evaluation**
- Checked collinearity between features used for modeling.
<img src="images/heatmap_time.jpg" alt="Heatmap of timebased features" width="550">
- Prepared data for statistical modeling by adjusting for the skewness and outliers of sales by using winsorization and the logarithm of sales.
- Aggregated features by month and split data into training and testing dataframes.
- Created forecasts for the past 6 months with and without exogenous variables using ARIMA, SARIMA and Holt-Winters.
<img src="images/arimax.jpg" alt="ARIMAX forecast" width="700">
<img src="images/sarimax.jpg" alt="SARIMAX forecast" width="700">
<img src="images/holt-winters.jpg" alt="Holt-Winters forecast" width="700">
- Evaluated each model on evaluation metrics such as mean absolute percentage error. <br><br>

**Model Interpretation**
- Acquired the residuals from the best-performing model's forecast vs the actual historical sales values.
<img src="images/sarimax_residuals.jpg" alt="Residuals from SARIMAX model" width="600">
- Visualized the best-performing model's forecast with a 95% confidence interval for the past 6 months.
<img src="images/sarimax_confidence.jpg" alt="SARIMAX forecast with confidence intervals" width="700">
- Compared the cumulative forecasted sales against the cumulative actual sales.
<img src="images/cumulative.jpg" alt="Cumulative actual sales vs SARIMAX sales" width="700"><br>

**Model Deployment**
- Developed a [Streamlit App](https://superstore-time-series-sales-forecasting.streamlit.app/) with interactive total sales filtering and forecasting.
- Saved sales data and generated freezed package versions of dependent packages for the streamlit app to avoid intercompatibility errors.<br><br>

## Insights
- More than 75% of sales have a value of less than 210 dollars, with the highest value being more than 22 thousand.
- The sales tend to be the highest in November and the lowest in February.
- The consumer segment is by far the greatest contributor to sales.
- The SARIMAX model has been found to be the best model with a MAPE at about 16%, producing fairly accurate forecasts.<br><br>

## Improvements
- Other statistical models such as quantile regression or bayesian structural time series could be used, which may be better at handling the skewness and outliers present in the sales data.
- Other techniques could also be employed to address the sales data's skewness and outliers such as robust scaling or box-cox transformation.
- Acquiring more data such as holidays, where the sales seem to spike, could also be beneficial. 
