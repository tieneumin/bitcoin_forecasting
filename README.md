# Bitcoin Price Direction Forecasting Tool

## Project Overview

This capstone project analyzes the relationship between Bitcoin market sentiment and close prices. The goal was to predict the subsequent day's (t+1) close price direction (i.e. bullish or bearish) by using traditional financial indicators from the current day (t) and day before (t-1), as well as sentiment metrics from external sources, in this case the Fear & Greed Index.

## Tools & Libraries Used

- **Python**
- [**yfinance**](https://pypi.org/project/yfinance/) – for fetching historical Bitcoin price data
- [**Fear & Greed Index**](https://alternative.me/crypto/fear-and-greed-index/) – as a sentiment indicator
- **Pandas, NumPy** – for data cleaning and preprocessing
- **Seaborn, Plotly** - for visualization of candlestick chart and time series decomposition
- [**Scikit-learn**](https://scikit-learn.org/) – for machine learning classification model training
- [Streamlit](https://streamlit.io/) - to present data in a user-friendly frontend

## Data Sources

- **Bitcoin Prices** - pulled using `yfinance` with the ticker `BTC-USD`
- **Fear & Greed Index** - pulled via API to provide a sentiment score ranging from 0 (Extreme Fear) to 100 (Extreme Greed); for more info, see Fear & Greed.md

## Methodology

1. **Data collection**:

   - Retrieved historical Bitcoin price data by day using `yfinance`
   - Collected corresponding Fear & Greed Index values

2. **Feature engineering and data analysis**:

   - Merged datasets by date, handling null values
   - Engineered price-based indicators (e.g. SMAs, EMAs, RSIs)
   - Visualized data in the form of a candlestick chart and conducted time series analysis to better understand the data

3. **Model training**:

   - Created binary `target` that is bullish (1) or bearish (0) by comparing close prices of current (t) and subsequent days (t+1)
   - Trained classifiers (e.g. Logistic Regression, Random Forest Classifier) to predict price direction, taking care not to introduce data leakage (e.g. training model on future data)

4. **Model evaluation**:
   - Evaluated models by accuracy, precision, recall, F1- and ROC-AUC scores
   - Extracted and sorted key features by importance

## Key Outcomes

- Demonstrated a link between the Fear & Greed Index and Bitcoin price movements
- Built a classification model capable of predicting the subsequent day's close price with relative accuracy
- Showcased methodology, model and conclusion in Streamlit

## Future Plans

- To enhance the model with additional sentiment sources (e.g. X (formerly Twitter), Reddit)
- Further explore time-series models or employ deep learning to improve prediction accuracy
- Implement a pipeline and dashboard for real-time prediction

## Repository Structure

```bash
bitcoin_forecasting/
├── datasets/                     # Raw and processed datasets
├── streamlit/                    # `streamlit run app.py` to view project
    ├── assets/                   # Trained model and images
├── Bitcoin Forecasting.ipynb     # Source code (data preprocessing & analysis, model training & evaluation)
├── Fear & Greed.md               # Explanation of the Fear & Greed Index as per https://alternative.me/crypto/fear-and-greed-index/
└── README.md                     # Project documentation
```
