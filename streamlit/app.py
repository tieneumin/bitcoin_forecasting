import streamlit as st
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
import pickle
import datetime

@st.cache_data
def load_data():
    pair_df = pd.read_csv('../datasets/BTC-USD.csv')
    fng_df = pd.read_csv('../datasets/fear_and_greed_index.csv')
    merged_df = pd.read_csv('../datasets/merged_dataset_with_features.csv')
    return [pair_df, fng_df, merged_df]

def statement():
    img1 = Image.open('assets/Bitcoin.png').resize((300, 300))
    img2 = Image.open('assets/OneDoesNotSimply.jpg').resize((450, 300))

    st.header('Problem Statement')
    
    col1, col2 = st.columns([2,3])
    with col1:
        st.image(img1)
    with col2:
        st.image(img2)
    st.markdown("As the largest cryptocurrency by market cap, Bitcoin's price fluctuations have historically posed a challenge to investors. In recent years, several cryptocurrencies including Bitcoin have seen significant growth, leading to an increase in the currency's volatility. To facilitate informed investment decisions, there is a growing need for robust time series forecasting models capable of making accurate predictions in light of the currency's seemingly random changes in price.")
    st.markdown('This application explores variables and factors that could contribute to better Bitcoin price predictions. An example model with several features can be found on the _Predict Price Direction_ page.')
    
def datasets():
    dfs = load_data()

    st.header('Datasets')
    st.markdown('Data was acquired from [Yahoo Finance](https://finance.yahoo.com/quote/BTC-USD/history/) and [Fear & Greed Index](https://alternative.me/crypto/fear-and-greed-index/) by means of API.')
    
    col1, col2 = st.columns([3,2])
    with col1:
        st.subheader('Yahoo Finance (BTC-USD)')
        slider1 = st.slider('Rows to display', 5, len(dfs[0]))
        st.dataframe(dfs[0].head(slider1))
    with col2:
        st.subheader('Fear & Greed Index')
        slider2 = st.slider('Rows to display', 5, len(dfs[1]))
        st.dataframe(dfs[1].head(slider2))

    st.subheader('Merged Dataset with Features')
    st.markdown('Datasets were merged on Fear & Greed Index as it had the shorter timeframe (2018-02-01 to present). BTC-USD OHLCV features were engineered pre-merge to mitigate null values.')
    slider3 = st.slider('Rows to display', 5, len(dfs[2]), key='slider3')
    st.dataframe(dfs[2].head(slider3))

def analysis():
    dfs = load_data()
    df = dfs[2]

    st.header('Exploratory Data Analysis')
    st.subheader('Descriptive Statistics')
    st.dataframe(dfs[2].describe().iloc[1:,0:])

    st.subheader('Visualizations')
    # candlestick chart
    fig = go.Figure(data=[go.Candlestick(
        x=df['date'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Candlestick'
    )])
    fig.add_trace(go.Scatter(
        x=df[df['target'] == 1]['date'],
        y=df[df['target'] == 1]['high'] + 2500,
        mode='markers',
        marker=dict(color='green', symbol='triangle-up', size=7),
        name='1: Close(t+1) > Close(t)',
        hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=df[df['target'] == 0]['date'],
        y=df[df['target'] == 0]['low'] - 2500,
        mode='markers',
        marker=dict(color='red', symbol='triangle-down', size=7),
        name='0: Close(t+1) < Close(t)',
        hoverinfo='skip'
    ))
    fig.update_layout(
        title='Candlestick Chart of Daily Bitcoin Close Price with Target Annotations',
        yaxis_title='Price (USD)',
        xaxis_title='Date'
    )
    st.plotly_chart(fig)

    # seasonal decomposition
    df2 = df.copy()
    df2['date'] = pd.to_datetime(df2['date'])
    df2.set_index('date',inplace=True)
    daily_data = df2['close'].resample('D').sum()
    result = seasonal_decompose(daily_data)

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=['Observed', 'Trend', 'Seasonal', 'Residual'])
    fig.add_trace(go.Scatter(x=daily_data.index, y=result.observed, mode='lines', name='Observed'), row=1, col=1)
    fig.add_trace(go.Scatter(x=daily_data.index, y=result.trend, mode='lines', name='Trend'), row=2, col=1)
    fig.add_trace(go.Scatter(x=daily_data.index, y=result.seasonal, mode='lines', name='Seasonal'), row=3, col=1)
    fig.add_trace(go.Scatter(x=daily_data.index, y=result.resid, mode='lines', name='Residual'), row=4, col=1)
    fig.update_yaxes(title_text='Price (USD)')
    fig.update_xaxes(title_text='Date', row=4)
    fig.update_layout(title_text='Seasonal Decomposition of Daily Bitcoin Close Price', showlegend=False)
    st.plotly_chart(fig)

    # heatmaps
    non_binary_df = pd.concat([
        df[['fng_value']],
        df.iloc[:,3:21],
        df.iloc[:,25:31],
    ], axis=1)
    plt.figure(figsize=(28,10))
    sns.heatmap(non_binary_df.corr(),annot=True,cmap='coolwarm',linewidths=0.5)
    plt.title('Correlation Heatmap of Non-Binary Features against Target')
    st.pyplot(plt)

    binary_df = pd.concat([df.iloc[:,21:26]], axis=1)
    plt.figure(figsize=(4,2))
    sns.heatmap(binary_df.corr(),annot=True,cmap='coolwarm',linewidths=0.5)
    plt.title('Correlation Heatmap of Binary Features against Target')
    st.pyplot(plt)

    # bar plot
    plt.figure()
    df['target'].value_counts().plot(kind='bar')
    plt.ylabel('Count')
    plt.xlabel('Target: Close(t+1) > Close(t)')
    plt.title('Bar Plot of Target Variable')
    plt.xticks([0,1], ['1 (True)','0 (False)'],rotation=0)
    st.pyplot(plt)

def model():
    with open('assets/LGBM_pipeline.pkl', 'rb') as file:
        pipeline = pickle.load(file)

    st.header('Predict Close Price Direction')
    st.markdown("Enter the following details to predict if the target day's (t+1) close price is greater or less than the prior day's (t) close price.")
    
    date = st.date_input('Date (t)', min_value=datetime.date(2018, 2, 1), max_value=datetime.date(2024, 12, 31))

    input_data = {
        'close_dir': st.selectbox('Close price (t) > Close price (t-1)? (1 = Yes, 0 = No)', options=[0, 1]),
        'high_dir': st.selectbox('Highest price (t) > Highest price (t-1)? (1 = Yes, 0 = No)', options=[0, 1]),
        'low_dir': st.selectbox('Lowest price (t) > Lowest price (t-1)? (1 = Yes, 0 = No)', options=[0, 1]),
        'volume_dir': st.selectbox('Volume (t) > Volume (t-1)? (1 = Yes, 0 = No)', options=[0, 1]),
        'rsi': st.number_input('Relative Strength Index (0-100)', min_value=0, max_value=100, value=50),
        'fng_value': st.number_input('[Fear & Greed Index](https://alternative.me/crypto/fear-and-greed-index/) (t)', min_value=5, max_value=95, value=50)
    }

    fng_prior = st.number_input('Fear & Greed Index (t-1)', min_value=5, max_value=95, value=50)
    input_data['fng_change'] = input_data['fng_value'] - fng_prior

    if input_data['fng_value'] > 75:
        input_data['fng_class'] = 'Extreme Greed'
    elif input_data['fng_value'] <= 75 and input_data['fng_value'] >= 55:
        input_data['fng_class'] = 'Greed'
    elif input_data['fng_value'] <= 54 and input_data['fng_value'] >= 47:
        input_data['fng_class'] = 'Neutral'
    elif input_data['fng_value'] <= 46 and input_data['fng_value'] >= 26:
        input_data['fng_class'] = 'Fear'      
    else:
        input_data['fng_class'] = 'Extreme Fear'

    input_data['year'] = date.year
    input_data['month'] = date.month
    input_data['day'] = date.day
    input_data['day_of_week'] = date.weekday()

    input_df = pd.DataFrame([input_data])
    prediction = pipeline.predict(input_df)
    
    st.subheader('Prediction')
    if prediction[0] == 1:
        st.markdown("Target day's (t+1) close price is expected to be greater than prior day's (t) close price.")
    elif prediction[0] == 0:
        st.markdown("Target day's (t+1) close price is expected to be less than prior day's (t) close price.")  
    else:
        st.markdown('An error occurred.')

st.set_page_config(
    page_title='Forecasting Bitcoin',
    page_icon='📈',
    layout='wide',
)
st.title('Bitcoin Price Forecasting App')

st.sidebar.header('Navigation')
pages = {
    'Problem Statement': statement,
    'View Datasets': datasets,
    'Exploratory Data Analysis': analysis,
    'Prediction': model,
}
page = st.sidebar.radio('Select a page', list(pages.keys()))
st.sidebar.info("This application predicts whether Bitcoin's close price for a certain day will be greater or less than the prior day's close price.")
st.sidebar.warning('Disclaimer: This model was developed for academic purposes and should not be used for real-world scenarios.')

pages[page]()


