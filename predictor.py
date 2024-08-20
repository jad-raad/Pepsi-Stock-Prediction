import requests
from bs4 import BeautifulSoup
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Function to scrape Pepsi stock data from Yahoo Finance
def scrape_pepsi_stock_data():
    symbol = 'PEP'
    url = f'https://finance.yahoo.com/quote/PEP?p=PEP&.tsrc=fin-srch'
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        stock_data = yf.download(symbol, start='2023-06-01', end='2024-01-02')  # Extended start date
        print("Scraped Data:")
        print(stock_data)  # Display the rows of scraped data
        return stock_data
    else:
        print(f"Failed to fetch the webpage. Status code: {response.status_code}")
        return None

# Function to incorporate political factor into the dataset
def add_political_factor(df):
    # Assume a binary political factor, where 1 represents a significant political event and 0 represents no event
    political_events = {
        '2023-10-07': 1,  # Example: Gaza War
        # Add more dates and corresponding political factors as needed
    }

    # Create a new column 'Political_Factor' and initialize with 0
    df['Political_Factor'] = 0

    # Set political factor to 1 on relevant dates
    for date, factor in political_events.items():
        if date in df.index:
            df.loc[date, 'Political_Factor'] = factor

# Scrape Pepsi stock data
pepsi_stock_data = scrape_pepsi_stock_data()

# Check if the data was successfully retrieved
if pepsi_stock_data is not None:
    # Feature engineering: Use 'Close' as the target variable and create lag features
    pepsi_stock_data['Target'] = pepsi_stock_data['Close'].shift(-1)
    pepsi_stock_data['Date'] = pepsi_stock_data.index

    # Add the political factor to the dataset
    add_political_factor(pepsi_stock_data)

    # Drop rows with missing values introduced by creating lag features
    pepsi_stock_data = pepsi_stock_data.dropna()

    # Select features and target variable
    features = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Political_Factor']
    X = pepsi_stock_data[features]
    y = pepsi_stock_data['Target']

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train a Random Forest regression model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions on the testing set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nModel Evaluation:")
    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')

    # Predict the next day's stock price
    last_data_point = X_scaled[-1].reshape(1, -1)
    next_day_prediction = model.predict(last_data_point)[0]

    print("\nPredicted Next Day Stock Price:")
    print(f"Last Data Point:\n{last_data_point}")
    print(f'Prediction: {next_day_prediction}')
else:
    print("No Pepsi stock data available.")
