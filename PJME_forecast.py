%%time
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import math

def create_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Create time series features based on the 'Datetime' column using Polars.
    """
    # Add lags
    for i in [1,2, 24, 48, 168]:
        df = df.with_columns(pl.col("PJME_MW").shift(i).alias(f"lag_{i}"))
    df = df.drop_nulls()
    
    return df.with_columns([
        pl.col("Datetime").dt.hour().alias("hour"),
        pl.col("Datetime").dt.weekday().alias("dayofweek"),
        pl.col("Datetime").dt.quarter().alias("quarter"),
        pl.col("Datetime").dt.month().alias("month"),
        pl.col("Datetime").dt.year().alias("year"),
        pl.col("Datetime").dt.ordinal_day().alias("dayofyear"),
        pl.col("Datetime").dt.day().alias("dayofmonth"),
        pl.col("Datetime").dt.week().alias("week")
    ])

def main():
    # Load dataset
    print("Loading data with Polars...")
    # Assume data is in the same directory or Kaggle input path
    try:
        df = pl.read_csv('Data/PJME_hourly.csv')
    except FileNotFoundError:
        # For Kaggle environment
        df = pl.read_csv('/kaggle/input/pjm-hourly-energy-consumption-data/PJME_hourly.csv')

    df = df.with_columns(pl.col("Datetime").str.to_datetime()).sort("Datetime")

    # Create features
    print("Creating features...")
    df = create_features(df)
    df.write_csv("Final.csv")

    # Train/Test Split
    split_date = pl.datetime(2015, 1, 1)
    train = df.filter(pl.col("Datetime") < split_date)
    test = df.filter(pl.col("Datetime") >= split_date)
    n = len(test)
    # print(f"Size of test is {n}")
    first_half_size = math.ceil(n / 2)  # n//2 + 1 if odd, n//2 if even

    val = test.slice(0, first_half_size)
    test = test.slice(first_half_size)
    
    FEATURES = ['hour', 'dayofweek', 'quarter', 'month', 'year', 'dayofyear', 'dayofmonth', 'week']
    FEATURES += [f"lag_{i}" for i in [1,2,24, 48, 168]]
        
    TARGET = 'PJME_MW'

    # Convert to NumPy for XGBoost
    X_train = train.select(FEATURES).to_numpy()
    y_train = train.select(TARGET).to_numpy().flatten()

    X_val, y_val = val.select(FEATURES).to_numpy(), val.select(TARGET).to_numpy().flatten()
    X_test = test.select(FEATURES).to_numpy()
    y_test = test.select(TARGET).to_numpy().flatten()
    # print(X_test.shape, y_test.shape, test.shape, X_val.shape, y_val.shape)
    # Model definition and training
    print("Training XGBoost model...")
    reg = xgb.XGBRegressor(
        base_score=0.5, 
        booster='gbtree',    
        # n_estimators=1000,
        n_estimators=5000,
        early_stopping_rounds=50,
        objective='reg:squarederror',
        max_depth=3,
        # learning_rate=0.01,
        learning_rate=0.05,
        # device='cuda'
    )
    
    reg.fit(X_train, y_train,
            # eval_set=[(X_train, y_train), (X_test, y_test)],
            eval_set=[(X_train, y_train), (X_val, y_val)],
            # eval_set=[ (X_test, y_test)],
            # eval_set=[(X_train, y_train)],
            verbose=100)

    # Forecasting
    print("Generating forecasts...")
    test_preds = reg.predict(X_test)
    test = test.with_columns(pl.Series(name="prediction", values=test_preds))

    # Metrics
    actual = test.select(TARGET).to_numpy().flatten()
    preds = test.select("prediction").to_numpy().flatten()
    
    rmse = np.sqrt(mean_squared_error(actual, preds))
    mae = mean_absolute_error(actual, preds)
    mape = mean_absolute_percentage_error(actual, preds)

    print(f'RMSE: {rmse:.2f}')
    print(f'MAE: {mae:.2f}')
    print(f'MAPE: {mape:.2%}')

    # Visualizations
    print("Saving visualizations...")
    
    # For plotting we can use the full dataframe
    # We'll convert to Pandas just for plotting convenience if needed, 
    # but let's try to stay Polars-native as much as possible by extracting what we need.
    
    full_dates = df.select("Datetime").to_numpy().flatten()
    full_actual = df.select(TARGET).to_numpy().flatten()
    test_dates = test.select("Datetime").to_numpy().flatten()
    test_preds_plot = test.select("prediction").to_numpy().flatten()

    # 1. Prediction vs Actual
    plt.figure(figsize=(15, 5))
    plt.plot(full_dates, full_actual, label='Actual', alpha=0.5)
    plt.plot(test_dates, test_preds_plot, label='Prediction', color='red', alpha=0.8)
    plt.title('PJM Energy Consumption Forecast (XGBoost + Polars)')
    plt.xlabel('Date')
    plt.ylabel('MW')
    plt.legend()
    plt.savefig('forecast_results.png')
    
    # 2. Zoomed in view (last month of testing)
    plt.figure(figsize=(15, 5))
    last_month_test = test.tail(24*30)
    plt.plot(last_month_test.select("Datetime").to_numpy().flatten(), 
             last_month_test.select(TARGET).to_numpy().flatten(), label='Actual')
    plt.plot(last_month_test.select("Datetime").to_numpy().flatten(), 
             last_month_test.select("prediction").to_numpy().flatten(), label='Prediction')
    plt.title('Zoomed View: Last Month of Testing')
    plt.legend()
    plt.savefig('forecast_zoomed.png')

    # 3. Feature Importance
    plt.figure(figsize=(10, 8))
    # importances = reg.feature_importances_
    sorted_idx = np.argsort(reg.feature_importances_)
    plt.barh(np.array(FEATURES)[sorted_idx], reg.feature_importances_[sorted_idx])
    # plt.barh(FEATURES, importances)
    plt.title('Feature Importance')
    plt.savefig('feature_importance.png')

    print("Done! Files saved: forecast_results.png, forecast_zoomed.png, feature_importance.png")

if __name__ == "__main__":
    main()
