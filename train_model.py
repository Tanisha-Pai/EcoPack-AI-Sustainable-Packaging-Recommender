import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
import joblib
import numpy as np

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
df = pd.read_csv("ecopack_kaggle_labeled.csv")

# -------------------------------------------------
# FEATURE ENGINEERING
# -------------------------------------------------
df["Volume_cm3"] = df["Length_cm"] * df["Width_cm"] * df["Height_cm"]

# -------------------------------------------------
# FEATURES & TARGET
# -------------------------------------------------
features = [
    "Warehouse_block",
    "Mode_of_Shipment",
    "Cost_of_the_Product",
    "Weight_in_gms",
    "Product_category",
    "Volume_cm3",
    "Fragile",
    "Liquid_Leak_Prone",
    "Perishable",
    "Shipping_distance",
    "Stacking_pressure",
    "Delivery_urgency",
    "Budget_level",
    "Eco_preference",
    "Material_constraint",
    "Return_risk",
    "Product_importance"
]

target = "recommended_packaging"

# -------------------------------------------------
# ENCODERS
# -------------------------------------------------
encoders = {}

# Warehouse encoder (forced classes)
warehouse_le = LabelEncoder()
warehouse_le.classes_ = np.array(["A", "B", "C", "D", "E", "F"])
df["Warehouse_block"] = warehouse_le.transform(df["Warehouse_block"])
encoders["Warehouse_block"] = warehouse_le

# Shipment encoder
ship_le = LabelEncoder()
ship_le.classes_ = np.array(["Flight", "Ship", "Road"])
df["Mode_of_Shipment"] = ship_le.transform(df["Mode_of_Shipment"])
encoders["Mode_of_Shipment"] = ship_le

# Other categorical encoders
categorical_cols = [
    "Product_category",
    "Shipping_distance",
    "Delivery_urgency",
    "Budget_level",
    "Eco_preference",
    "Material_constraint",
    "Product_importance"
]

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Target encoding
le_target = LabelEncoder()
df[target] = le_target.fit_transform(df[target])

# -------------------------------------------------
# 🔥 DATA BALANCING 
# -------------------------------------------------
df_bal = df[features + [target]]

# Find smallest class count
min_count = df_bal[target].value_counts().min()

# Undersample all classes equally
balanced_df = pd.concat([
    resample(group, replace=False, n_samples=min_count, random_state=42)
    for _, group in df_bal.groupby(target)
])

# Shuffle dataset
balanced_df = balanced_df.sample(frac=1, random_state=42)

# Split features and target
X = balanced_df[features]
y = balanced_df[target]

# -------------------------------------------------
# TRAIN TEST SPLIT
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------------------------
# MODEL TRAINING
# -------------------------------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -------------------------------------------------
# EVALUATION
# -------------------------------------------------
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

print("✅ KNN Model trained successfully!")
print("🎯 Accuracy:", round(acc * 100, 2), "%")

# -------------------------------------------------
# SAVE FILES
# -------------------------------------------------
joblib.dump(model, "random_forest_model.pkl")
joblib.dump(encoders, "feature_encoders.pkl")
joblib.dump(le_target, "target_encoder.pkl")

print("✅ Model & encoders saved!")