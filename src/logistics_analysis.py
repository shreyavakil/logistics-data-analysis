import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ----------------------------------------
# 1. LOAD DATA
# ----------------------------------------

orders = pd.read_csv("data/orders.csv")


print("Dataset Shape:", orders.shape)
print(orders.head())


# ----------------------------------------
# 2. DATA CLEANING
# ----------------------------------------

# Remove duplicate records
orders = orders.drop_duplicates()

# Convert date columns
orders["order_date"] = pd.to_datetime(
    orders["order_date"]
)

orders["actual_delivery"] = pd.to_datetime(
    orders["actual_delivery"]
)

orders["promised_delivery"] = pd.to_datetime(
    orders["promised_delivery"]
)


# ----------------------------------------
# 3. CALCULATE DELIVERY DELAY
# ----------------------------------------

orders["delay_minutes"] = (
    orders["actual_delivery"]
    - orders["promised_delivery"]
).dt.total_seconds() / 60


# ----------------------------------------
# 4. KPI CALCULATION
# ----------------------------------------

# On-Time Delivery Rate
on_time_rate = (
    orders["delay_minutes"] <= 0
).mean() * 100

# Average delivery delay
average_delay = orders["delay_minutes"].mean()

# Average delivery distance
average_distance = orders["distance_km"].mean()

print("\n--- LOGISTICS KPIs ---")

print(
    "On-Time Delivery Rate:",
    round(on_time_rate, 2),
    "%"
)

print(
    "Average Delivery Delay:",
    round(average_delay, 2),
    "minutes"
)

print(
    "Average Delivery Distance:",
    round(average_distance, 2),
    "km"
)


# ----------------------------------------
# 5. EXPLORATORY DATA ANALYSIS
# ----------------------------------------

warehouse_delay = (
    orders.groupby("warehouse_id")["delay_minutes"]
    .mean()
)

print("\nAverage Delay by Warehouse:")
print(warehouse_delay)


warehouse_delay.plot(
    kind="bar",
    title="Average Delivery Delay by Warehouse"
)

plt.xlabel("Warehouse")
plt.ylabel("Average Delay (Minutes)")
plt.tight_layout()
plt.show()


# ----------------------------------------
# 6. MACHINE LEARNING
# Predict Delivery Delay
# ----------------------------------------

features = [
    "distance_km",
    "order_quantity",
    "vehicle_capacity"
]

X = orders[features].fillna(0)

y = orders["delay_minutes"].fillna(0)


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# Train Random Forest model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)


# Make predictions
predictions = model.predict(X_test)


# ----------------------------------------
# 7. MODEL EVALUATION
# ----------------------------------------

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

print("\n--- MODEL PERFORMANCE ---")

print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))


# ----------------------------------------
# 8. FEATURE IMPORTANCE
# ----------------------------------------

importance = pd.Series(
    model.feature_importances_,
    index=features
).sort_values(
    ascending=False
)

print("\nFeature Importance:")
print(importance)
