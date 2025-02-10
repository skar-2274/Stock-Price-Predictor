import yfinance as yf
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import pickle
import sys

# Data Collection
def fetch_stock_data(symbol, start):
    try:
        print(f"Fetching data for {symbol} from {start} to today...")
        df = yf.download(symbol, start=start)

        if df.empty:
            raise ValueError(f"No data found for {symbol}.")
        
        print("Data successfully fetched!")

        # Data Preprocessing
        missing_count = df.isna().sum().sum()
        missing_threshold = int(0.05 * df.size)
        if missing_count > 0:
            if missing_count > missing_threshold :
                print(f"Warning: Data contains {missing_count} missing values")

            print("Interpolating missing values...")
            df.interpolate(method="linear", inplace=True)

            if df.isna().sum().sum() > 0:
                print("Filling missing values using forward and backward fill...")
                df.ffill(inplace=True)
                df.bfill(inplace=True)

        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

# Feature Engineering/Indicators
SMA_PERIOD = 200

def add_sma(df):
    df[f"SMA_{SMA_PERIOD}"] = df["Close"].rolling(window=SMA_PERIOD).mean()
    df["Target"] = df["Close"].shift(-1)
    df.dropna(inplace=True)
    return df

# Train-Test Split
def split_data(df):
    X = df[[f"SMA_{SMA_PERIOD}"]]
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    return X_train, X_test, y_train, y_test

# Hyperparameter Tuning using RandomizedSearchCV
def tune_xgboost_random(X_train, y_train):
    param_dist = {
        'n_estimators': [50, 100, 200, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
        'max_depth': [3, 5, 7, 10],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }

    model = XGBRegressor(objective="reg:squarederror")

    random_search = RandomizedSearchCV(
        estimator=model, param_distributions=param_dist, n_iter=10,
        scoring='neg_mean_squared_error', n_jobs=-1, cv=3, verbose=1, random_state=42
    )

    random_search.fit(X_train, y_train)

    print("\n Best Hyperparameters (RandomizedSearchCV):", random_search.best_params_)
    return random_search.best_estimator_

# Regression Model Selector
def train_xgboost(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train) # Train Model

    y_pred = model.predict(X_test) # Predictions

    # Evaluate Model RMSE, MAE, R^2
    print("\nXGBoost Regression Performance:")
    print(f"RMSE: {root_mean_squared_error(y_test, y_pred):.2f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"R\u00b2 Score: {r2_score(y_test, y_pred):.2f}")

    return model

def main():

    if len(sys.argv) > 1:
        stock_symbol = sys.argv[1]
    else:
        stock_symbol = input("Enter Stock Ticker (e.g., AAPL, TSLA, MSFT): ") 

    start_date = "2020-01-01"

    df = fetch_stock_data(stock_symbol, start_date)
    
    if df is not None:
        df = add_sma(df)
        print(df.head())

        X_train, X_test, y_train, y_test = split_data(df)

        best_model = tune_xgboost_random(X_train, y_train)

        trained_model = train_xgboost(best_model, X_train, X_test, y_train, y_test)

        # Model Deployment using Streamlit
        model_filename = f"stock_price_model_{stock_symbol}.pkl"
        with open(model_filename, "wb") as f:
            pickle.dump(trained_model, f)
        print(f"\nModel saved as '{model_filename}'")

if __name__ == "__main__":
    main()