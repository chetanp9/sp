import streamlit as st
import hashlib
import pymongo
import yfinance as yf
import plotly.graph_objs as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pymongo.server_api import ServerApi
import pandas as pd
import calendar

# MongoDB connection setup
mongo_uri = "mongodb+srv://chetanpal9826473966:jS9zwMr382ljiNB9@cluster0.ysistma.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = pymongo.MongoClient(mongo_uri, server_api=ServerApi("1"))
db = client["stock_trading_platform"]
users_collection = db["users"]

# Function to hash passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Function to authenticate user
def authenticate_user(username, password):
    user = users_collection.find_one({"username": username})
    if user and user["password"] == hash_password(password):
        return True, user
    return False, None

# Function to register a new user
def register_user(username, password):
    hashed_password = hash_password(password)
    user_data = {"username": username, "password": hashed_password, "watchlist": []}
    users_collection.insert_one(user_data)
    st.success("Registration successful!")

# Function to add stock to the watchlist
WATCHLIST_MAX_SIZE = 10

def add_to_watchlist(username, stock_symbol):
    user = users_collection.find_one({"username": username})
    watchlist = user.get("watchlist", [])

    if stock_symbol in watchlist:
        return f"{stock_symbol} is already in your watchlist."

    if len(watchlist) >= WATCHLIST_MAX_SIZE:
        return "Watchlist has reached its maximum capacity. Please remove a stock before adding a new one."

    watchlist.append(stock_symbol)
    users_collection.update_one({"username": username}, {"$set": {"watchlist": watchlist}})
    return f"Added {stock_symbol} to your watchlist."

# Function to remove stock from the watchlist
def remove_from_watchlist(username, stock_symbol):
    user = users_collection.find_one({"username": username})
    watchlist = user.get("watchlist", [])

    if stock_symbol not in watchlist:
        return f"{stock_symbol} is not in your watchlist."

    watchlist.remove(stock_symbol)
    users_collection.update_one({"username": username}, {"$set": {"watchlist": watchlist}})
    return f"Removed {stock_symbol} from your watchlist."

# Initialize session state
def initialize_session():
    if "user_authenticated" not in st.session_state:
        st.session_state["user_authenticated"] = False
    if "username" not in st.session_state:
        st.session_state["username"] = None
    if "user_data" not in st.session_state:
        st.session_state["user_data"] = None

# Set user session state
def set_user_session(authenticated, username=None, user_data=None):
    st.session_state["user_authenticated"] = authenticated
    st.session_state["username"] = username if authenticated else None
    st.session_state["user_data"] = user_data if authenticated else None

# Streamlit UI for login
def login():
    st.title("Login")
    st.image('icons8-trading-100.png', width=100)
    st.write("Welcome to the trading platform. Please log in to continue.")

    # Login form
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        authenticated, user_data = authenticate_user(username, password)
        if authenticated:
            set_user_session(True, username, user_data)  # Store user data in session
            return username
        else:
            st.error("Invalid username or password")
    return None

# Streamlit UI for registration
def registration():
    st.title("Registration")
    st.image('icons8-registration-64.png', width=100)
    st.write("Welcome to the trading platform. Please register to continue.")

    # Registration form
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")

    if st.button("Register"):
        if password != confirm_password:
            st.error("Passwords do not match")
        elif len(password) < 8:
            st.error("Password must be at least 8 characters long")
        else:
            register_user(username, password)

# Streamlit UI for the main app with watchlist functionality
def main_app():
    st.title("Stock Price Prediction with Technical Indicators")
    st.image('icons8-namaste-78.png', width=100)

    if st.session_state["user_authenticated"]:
        user_data = st.session_state["user_data"]
        username = user_data["username"]
        watchlist = user_data.get("watchlist", [])

        st.write(f"Welcome, {username}!")

        # Logout button
        if st.button("Logout"):
            set_user_session(False)
            st.experimental_set_query_params(login="false")
            return

        # Sidebar for watchlist and stock analysis
        st.sidebar.header("Your Watchlist")

        if watchlist:
            selected_stock = st.sidebar.selectbox("Select a stock from your watchlist", ["-"] + watchlist)
            if selected_stock != "-":
                stock_symbol = selected_stock
            else:
                stock_symbol = st.sidebar.text_input("Enter stock symbol", 'GOOG')
        else:
            stock_symbol = st.sidebar.text_input("Enter stock symbol", 'GOOG')

        # Add stock to watchlist
        if st.sidebar.button("Add to Watchlist"):
            add_message = add_to_watchlist(username, stock_symbol)
            if "Added" in add_message:
                st.sidebar.success(add_message)
            else:
                st.sidebar.warning(add_message)

        # Remove stock from watchlist
        if st.sidebar.button("Remove from Watchlist"):
            remove_message = remove_from_watchlist(username, stock_symbol)
            if "Removed" in remove_message:
                st.sidebar.success(remove_message)
            else:
                st.sidebar.warning(remove_message)

        # Stock analysis parameters
        start_date = st.sidebar.text_input("Enter start date (YYYY-MM-DD)", '2010-01-01')
        end_date = st.sidebar.text_input("Enter end date (YYYY-MM-DD)", '2022-01-01')
        chart_type = st.sidebar.selectbox("Select Chart Type", ['Line Chart', 'Candlestick Chart'])

        if st.sidebar.button("Train Model"):
            data = download_stock_data(stock_symbol, start_date, end_date)
            data_with_indicators = calculate_technical_indicators(data)
            model = train_prediction_model(data_with_indicators)
            future_end_date = (pd.to_datetime(end_date) + pd.DateOffset(months=1))
            future_month_name = calendar.month_name[future_end_date.month]
            future_data = download_stock_data(stock_symbol, end_date, future_end_date)
            future_data_with_indicators = calculate_technical_indicators(future_data)
            predicted_prices = predict_prices(model, future_data_with_indicators)

            # Calculate the range of predicted prices
            min_pred_price = predicted_prices.min()
            max_pred_price = predicted_prices.max()
            pred_price_range = f"${min_pred_price:.2f} - ${max_pred_price:.2f}"

            # Display results and plot
            st.subheader("Stock Data with Technical Indicators and Predictions")
            st.dataframe(data_with_indicators)
            st.subheader(f"Predicted Prices for {future_month_name}")
            st.write("Predicted Price Range:", pred_price_range)
            st.write("predicted Price Data:",predicted_prices)

            # Plot technical indicators and predictions
            fig = go.Figure()

            # Plot Closing Price
            if chart_type == 'Candlestick Chart':
                fig.add_trace(go.Candlestick(x=data_with_indicators.index,
                                                open=data_with_indicators['Open'],
                                                high=data_with_indicators['High'],
                                                low=data_with_indicators['Low'],
                                                close=data_with_indicators['Close'],
                                                name='Candlestick'))
            else:
                fig.add_trace(go.Scatter(x=data_with_indicators.index, y=data_with_indicators['Close'], name='Close'))

            # Plot EMA, MACD, and RSI
            fig.add_trace(go.Scatter(x=data_with_indicators.index, y=data_with_indicators['EMA_12'], name='EMA 12'))
            fig.add_trace(go.Scatter(x=data_with_indicators.index, y=data_with_indicators['EMA_26'], name='EMA 26'))

            fig.add_trace(go.Scatter(x=data_with_indicators.index, y=data_with_indicators['MACD'], name='MACD'))
            fig.add_trace(go.Scatter(x=data_with_indicators.index, y=data_with_indicators['MACD_Signal'], name='MACD Signal'))

            fig.add_trace(go.Scatter(x=data_with_indicators.index, y=data_with_indicators['RSI'], name='RSI'))

            # Predicted Prices
            #fig.add_trace(go.Scatter(x=data_with_indicators.index, y=predicted_prices, mode='lines', name='Predicted Prices'))

            # Update layout and display plot
            fig.update_layout(title='Stock Price and Technical Indicators',
                              xaxis_title='Date',
                              yaxis_title='Value',
                              width=1000,
                              height=600,
                              xaxis=dict(
                                  rangeselector=dict(
                                      buttons=[
                                          dict(count=1, label='1m', step='month', stepmode='backward'),
                                          dict(count=6, label='6m', step='month', stepmode='backward'),
                                          dict(count=1, label='YTD', step='year', stepmode='todate'),
                                          dict(count=1, label='1y', step='year', stepmode='backward'),
                                          dict(step='all')
                                      ]),
                                  rangeslider=dict(visible=True),
                                  type='date'),
                              yaxis=dict(fixedrange=False),
                              dragmode='zoom') 

            # Display the plot
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Please login to access the stock trading platform.")

# Download stock data
def download_stock_data(stock_symbol, start_date, end_date):
    try:
        data = yf.download(stock_symbol, start=start_date, end=end_date)
        return data
    except Exception as e:
        st.error(f"Error occurred while downloading data: {e}")
        return None

# Calculate technical indicators
def calculate_technical_indicators(data):
    data["EMA_12"] = data["Close"].ewm(span=12, adjust=False).mean()
    data["EMA_26"] = data["Close"].ewm(span=26, adjust=False).mean()

    data["MACD"] = data["EMA_12"] - data["EMA_26"]
    data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()

    delta = data["Close"].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()

    rs = gain / loss
    data["RSI"] = 100 - (100 / (1 + rs))

    return data

# Train the prediction model
def train_prediction_model(data):
    X = data[["EMA_12", "EMA_26", "MACD", "MACD_Signal", "RSI"]]
    y = data["Close"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    st.write("Mean Squared Error:", mse)
    st.write("Mean Absolute Error:", mae)

    return model

# Make predictions
def predict_prices(model, data):
    if data.empty:
        return []

    X_pred = data[["EMA_12", "EMA_26", "MACD", "MACD_Signal", "RSI"]]
    predictions = model.predict(X_pred)
    return predictions

# Main function to start the app
def main():
    initialize_session()

    st.set_page_config(page_title="Stock Trading Platform", page_icon=":chart_with_upwards_trend:")
    st.sidebar.title("Navigation")

    user_authenticated = st.session_state["user_authenticated"]

    if not user_authenticated:
        choice = st.sidebar.radio("Go to", ("Login", "Registration"))
        if choice == "Login":
            username = login()
            if username:
                st.experimental_set_query_params(login="true")
        elif choice == "Registration":
            registration()

    if st.session_state["user_authenticated"]:
        main_app()

if __name__ == "__main__":
    main()
