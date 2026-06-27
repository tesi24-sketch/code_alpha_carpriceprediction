# ============================================================
#  TASK 3: Car Price Prediction with Machine Learning
#  CodeAlpha Internship
# ============================================================

# ── IMPORTS ──────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# STEP 1 ▸ LOAD DATASET
# ─────────────────────────────────────────────────────────────
# ► Use this after downloading the dataset from Kaggle:
df = pd.read_csv("car data.csv")          # or whatever filename you have
#
# ► Synthetic dataset (mirrors real car dataset structure):
np.random.seed(42)
n = 300

brands = ["Maruti", "Honda", "Toyota", "Hyundai", "Ford", "BMW", "Audi", "Tata"]
fuel_types = ["Petrol", "Diesel", "CNG"]
seller_types = ["Dealer", "Individual"]
transmissions = ["Manual", "Automatic"]

df = pd.DataFrame({
    "Car_Name":      np.random.choice(brands, n),
    "Year":          np.random.randint(2005, 2023, n),
    "Selling_Price":  None,                                # target — filled below
    "Present_Price": np.random.uniform(3, 20, n).round(2),
    "Kms_Driven":    np.random.randint(5000, 150000, n),
    "Fuel_Type":     np.random.choice(fuel_types, n),
    "Seller_Type":   np.random.choice(seller_types, n),
    "Transmission":  np.random.choice(transmissions, n),
    "Owner":         np.random.choice([0, 1, 2, 3], n),
})

# Realistic price generation
brand_goodwill = df["Car_Name"].map(
    {"Maruti": 1, "Honda": 1.3, "Toyota": 1.4, "Hyundai": 1.2,
     "Ford": 1.1, "BMW": 2.5, "Audi": 2.3, "Tata": 0.9}
)
age = 2024 - df["Year"]
df["Selling_Price"] = (
        df["Present_Price"] * brand_goodwill
        - 0.05 * age
        - 0.000015 * df["Kms_Driven"]
        - 0.4 * df["Owner"]
        + np.random.normal(0, 0.5, n)
).clip(0.5).round(2)

print("=" * 55)
print("  DATASET OVERVIEW")
print("=" * 55)
print(df.head(10).to_string(index=False))
print(f"\nShape  : {df.shape}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nMissing Values:\n{df.isnull().sum()}")
print(f"\nDescriptive Statistics:\n{df.describe().round(2)}")


# ─────────────────────────────────────────────────────────────
# STEP 2 ▸ EXPLORATORY DATA ANALYSIS (EDA)
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("EDA – Car Price Dataset", fontsize=14, fontweight="bold")

# Selling Price distribution
axes[0][0].hist(df["Selling_Price"], bins=30, color="#457b9d", edgecolor="white", alpha=0.85)
axes[0][0].set_title("Distribution of Selling Price")
axes[0][0].set_xlabel("Price (Lakhs)")

# Present Price vs Selling Price
axes[0][1].scatter(df["Present_Price"], df["Selling_Price"],
                   alpha=0.5, color="#2a9d8f", edgecolors="white", s=40)
axes[0][1].set_title("Present Price vs Selling Price")
axes[0][1].set_xlabel("Present Price")
axes[0][1].set_ylabel("Selling Price")

# Avg price by Brand
brand_avg = df.groupby("Car_Name")["Selling_Price"].mean().sort_values(ascending=False)
axes[0][2].bar(brand_avg.index, brand_avg.values, color="#e76f51", edgecolor="white", alpha=0.85)
axes[0][2].set_title("Avg Price by Brand")
axes[0][2].set_xticklabels(brand_avg.index, rotation=45, ha="right")
axes[0][2].set_ylabel("Avg Price")

# Kms Driven vs Price
axes[1][0].scatter(df["Kms_Driven"], df["Selling_Price"],
                   alpha=0.5, color="#f4a261", edgecolors="white", s=40)
axes[1][0].set_title("Kms Driven vs Selling Price")
axes[1][0].set_xlabel("Kms Driven")
axes[1][0].set_ylabel("Selling Price")

# Avg price by Fuel Type
fuel_avg = df.groupby("Fuel_Type")["Selling_Price"].mean()
axes[1][1].bar(fuel_avg.index, fuel_avg.values, color=["#e63946", "#2a9d8f", "#f4a261"],
               edgecolor="white", alpha=0.85)
axes[1][1].set_title("Avg Price by Fuel Type")
axes[1][1].set_ylabel("Avg Price")

# Avg price by Transmission
trans_avg = df.groupby("Transmission")["Selling_Price"].mean()
axes[1][2].bar(trans_avg.index, trans_avg.values, color=["#457b9d", "#a8dadc"],
               edgecolor="white", alpha=0.85)
axes[1][2].set_title("Avg Price by Transmission")
axes[1][2].set_ylabel("Avg Price")

plt.tight_layout()
plt.savefig("eda_car_price.png", dpi=120)
plt.show()

# Correlation Heatmap
numeric_cols = ["Year", "Present_Price", "Kms_Driven", "Owner", "Selling_Price"]
plt.figure(figsize=(7, 5))
sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f",
            cmap="coolwarm", linewidths=0.5, square=True)
plt.title("Correlation Heatmap", fontweight="bold")
plt.tight_layout()
plt.savefig("correlation_heatmap_car.png", dpi=120)
plt.show()


# ─────────────────────────────────────────────────────────────
# STEP 3 ▸ DATA PREPROCESSING & FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
# Feature Engineering — car age is more meaningful than year
df["Car_Age"] = 2024 - df["Year"]

# Encode categorical columns
le = LabelEncoder()
for col in ["Car_Name", "Fuel_Type", "Seller_Type", "Transmission"]:
    df[col + "_enc"] = le.fit_transform(df[col])

# Feature Selection
feature_cols = [
    "Car_Name_enc", "Car_Age", "Present_Price",
    "Kms_Driven", "Fuel_Type_enc", "Seller_Type_enc",
    "Transmission_enc", "Owner"
]
X = df[feature_cols]
y = df["Selling_Price"]

# Train / Test Split (80 / 20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Scale features (for Linear Regression)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"\nTraining samples : {X_train.shape[0]}")
print(f"Testing  samples : {X_test.shape[0]}")


# ─────────────────────────────────────────────────────────────
# STEP 4 ▸ MODEL TRAINING & EVALUATION
# ─────────────────────────────────────────────────────────────
def evaluate_model(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"\n{'─'*42}")
    print(f"  {name}")
    print(f"{'─'*42}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  R²   : {r2:.4f}")
    return {"Model": name, "MAE": mae, "RMSE": rmse, "R²": r2}

results = []

# ── Model 1: Linear Regression ───────────────────────────────
lr = LinearRegression()
lr.fit(X_train_sc, y_train)
y_pred_lr = lr.predict(X_test_sc)
results.append(evaluate_model("Linear Regression", y_test, y_pred_lr))

# ── Model 2: Random Forest ───────────────────────────────────
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
results.append(evaluate_model("Random Forest", y_test, y_pred_rf))

# ── Model 3: Gradient Boosting ───────────────────────────────
gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                               max_depth=4, random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
results.append(evaluate_model("Gradient Boosting", y_test, y_pred_gb))


# ─────────────────────────────────────────────────────────────
# STEP 5 ▸ MODEL COMPARISON
# ─────────────────────────────────────────────────────────────
results_df = pd.DataFrame(results).set_index("Model")
print("\n\n" + "=" * 55)
print("  MODEL COMPARISON SUMMARY")
print("=" * 55)
print(results_df.round(4).to_string())

best_model_name = results_df["R²"].idxmax()
print(f"\n  ★ Best Model : {best_model_name}  (R² = {results_df.loc[best_model_name,'R²']:.4f})")

# Bar chart comparison
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, metric, color in zip(axes, ["MAE", "RMSE", "R²"],
                             ["#e63946", "#457b9d", "#2a9d8f"]):
    bars = ax.bar(results_df.index, results_df[metric], color=color, alpha=0.85, edgecolor="white")
    ax.set_title(metric, fontweight="bold")
    ax.set_ylabel(metric)
    ax.set_xticklabels(results_df.index, rotation=10, ha="right")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
plt.suptitle("Model Comparison – Car Price Prediction", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("model_comparison_car.png", dpi=120)
plt.show()


# ─────────────────────────────────────────────────────────────
# STEP 6 ▸ FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────
importances = pd.Series(gb.feature_importances_, index=feature_cols).sort_values(ascending=True)

plt.figure(figsize=(8, 5))
importances.plot(kind="barh", color="#457b9d", edgecolor="white")
plt.title("Feature Importance – Gradient Boosting", fontweight="bold")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("feature_importance_car.png", dpi=120)
plt.show()

print("\nFeature Importance:")
for feat, imp in importances.sort_values(ascending=False).items():
    print(f"  {feat:<22} : {imp:.4f}")


# ─────────────────────────────────────────────────────────────
# STEP 7 ▸ ACTUAL vs PREDICTED
# ─────────────────────────────────────────────────────────────
plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred_gb, alpha=0.7, color="#2a9d8f", edgecolors="white", s=55)
lims = [min(y_test.min(), y_pred_gb.min()) - 0.5,
        max(y_test.max(), y_pred_gb.max()) + 0.5]
plt.plot(lims, lims, "r--", linewidth=1.5, label="Perfect Prediction")
plt.xlabel("Actual Price (Lakhs)")
plt.ylabel("Predicted Price (Lakhs)")
plt.title("Actual vs Predicted Car Price", fontweight="bold")
plt.legend()
plt.tight_layout()
plt.savefig("actual_vs_predicted_car.png", dpi=120)
plt.show()


# ─────────────────────────────────────────────────────────────
# STEP 8 ▸ PREDICT NEW CAR PRICE
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  SAMPLE PREDICTION")
print("=" * 55)

# Example: Honda, 2018, Present Price 8L, 40000 km, Petrol, Dealer, Manual, 0 owners
sample = pd.DataFrame({
    "Car_Name_enc":     [1],     # Honda encoded
    "Car_Age":          [6],     # 2024 - 2018
    "Present_Price":    [8.0],
    "Kms_Driven":       [40000],
    "Fuel_Type_enc":    [1],     # Petrol
    "Seller_Type_enc":  [0],     # Dealer
    "Transmission_enc": [0],     # Manual
    "Owner":            [0],
})

pred_lr = lr.predict(scaler.transform(sample))[0]
pred_rf = rf.predict(sample)[0]
pred_gb = gb.predict(sample)[0]

print(f"\n  Car     : Honda | Year: 2018 | 40,000 km | Petrol | Manual")
print(f"  Present Price : ₹8.00 Lakhs")
print(f"\n  Linear Regression  → ₹{pred_lr:.2f} Lakhs")
print(f"  Random Forest      → ₹{pred_rf:.2f} Lakhs")
print(f"  Gradient Boosting  → ₹{pred_gb:.2f} Lakhs")

print("\n✅ Car Price Prediction completed successfully!")