import yfinance as yf
import pandas as pd
import numpy as np

def get_data(tickers, start='2014-01-01', end='2026-01-01'):
    stock_data = {}
    for ticker in tickers: 
        df = yf.download(ticker, start=start, end=end) 
        stock_data[ticker] = df
    return stock_data

def split_data(stock_data, training_range=('2009-01-01', '2019-12-31'), 
                validation_range=('2020-01-01', '2020-12-31'),
                test_range=('2021-01-01', '2025-01-01')):

    training_data = {}
    validation_data = {}
    test_data = {}
    
    for ticker, df in stock_data.items():
        training_data[ticker] = df.loc[training_range[0]:training_range[1]]
        validation_data[ticker] = df.loc[validation_range[0]:validation_range[1]]
        test_data[ticker] = df.loc[test_range[0]:test_range[1]]

    return training_data, validation_data, test_data

# Technical Indicators
def RSI(df, window=14): 
    delta = df['Close'].diff()  
    up = delta.where(delta > 0, 0)   
    down = -delta.where(delta < 0, 0)   
    rs = up.rolling(window=window).mean() / down.rolling(window=window).mean()    
    return 100 - 100 / (1 + rs)   

def bollinger_bands(df): 
    sma = df['Close'].rolling(window=20).mean() 
    std = df['Close'].rolling(window=20).std()  
    upper_band = sma + 2 * std
    lower_band = sma - 2 * std 
    return upper_band, lower_band 

def relative_volume(df): 
    return df['Volume'] / df['Volume'].rolling(window=20).mean()

def add_technical_indicators(df):  
    df_new = df.copy()  # Avoid SettingWithCopyWarning
    df_new['RSI'] = RSI(df)
    upper_bb, lower_bb = bollinger_bands(df)
    df_new['Upper_BB'] = upper_bb
    df_new['Lower_BB'] = lower_bb
    df_new['EMA12'] = df_new['Close'].ewm(span=12, adjust=False).mean() 
    df_new['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean() 
    df_new['MACD'] = df_new['EMA12'] - df_new['EMA26']
    df_new['RVOL'] = relative_volume(df)
    df_new.dropna(inplace=True)
    return df_new


def process_data_with_indicators(stock_data):
    processed_data = {}
    for ticker, df in stock_data.items(): 
        processed_data[ticker] = add_technical_indicators(df) 
    return processed_data
