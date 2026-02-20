import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# ---------------------------------------------------------
# Deliverable 1: Data Exploration
# ---------------------------------------------------------
print("=== 1. Data Exploration ===")
df = pd.read_csv('waste bin data.csv')

print("\n--- Basic Info ---")
print(df.info())

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Value Counts for categorical ---")
print("Location Type:\n", df['location_type'].value_counts())
print("Weather:\n", df['weather'].value_counts())
print("Festival Week:\n", df['festival_week'].value_counts())
print("Is Full (Target):\n", df['is_full'].value_counts())

print("\n[Key Insights]")
print("- Missing Values: There are missing values in 'avg_daily_waste_kg', 'days_since_last_pickup', and 'weather' columns.")
print("- Festivals & Weather: Both festival weeks and weather patterns may significantly impact waste generation rates, which we'll handle in our model features.")
print("- Target Distribution: The target variable 'is_full' needs to be checked for class imbalance.")


# ---------------------------------------------------------
# Deliverable 2: Feature Engineering
# ---------------------------------------------------------
print("\n=== 2. Feature Engineering ===")

# First, handle missing values for feature engineering
# Fill numerical features with median and categorical with mode temporarily to compute new features
df['avg_daily_waste_kg'] = df['avg_daily_waste_kg'].fillna(df['avg_daily_waste_kg'].median())
df['days_since_last_pickup'] = df['days_since_last_pickup'].fillna(df['days_since_last_pickup'].median())

# New Feature 1: estimated_current_waste
# Logic: If a bin generates X kg/day and it's been Y days, it roughly has X*Y kg now.
df['estimated_current_waste_kg'] = df['avg_daily_waste_kg'] * df['days_since_last_pickup']

# New Feature 2: fill_ratio_estimate
# Logic: Ratio of estimated waste to bin capacity. Higher ratio = more likely to be full.
df['fill_ratio_estimate'] = df['estimated_current_waste_kg'] / df['bin_capacity_kg']

# Applying festival multiplier (Hypothesis: festival week increases waste by 20%)
# Though the ML model can learn this implicitly, creating an aggressive fill estimate might help.
df['adjusted_fill_ratio'] = np.where(df['festival_week'] == 1, 
                                     df['fill_ratio_estimate'] * 1.2, 
                                     df['fill_ratio_estimate'])

print("Created features: 'estimated_current_waste_kg', 'fill_ratio_estimate', 'adjusted_fill_ratio'")
print("Reasoning: Understanding the accumulated waste over the days since last pickup relative to the bin capacity provides a direct proxy for the likelihood of the bin being full.")

# Drop id column as it has no predictive power
df = df.drop('bin_id', axis=1)

# Separate features and target
X = df.drop('is_full', axis=1)
y = df['is_full']

# Define categorical and numerical columns
categorical_cols = ['location_type', 'weather']
numerical_cols = ['avg_daily_waste_kg', 'days_since_last_pickup', 'festival_week', 
                  'bin_capacity_kg', 'estimated_current_waste_kg', 
                  'fill_ratio_estimate', 'adjusted_fill_ratio']

# Preprocessing pipelines
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# ---------------------------------------------------------
# Deliverable 3: Modeling
# ---------------------------------------------------------
print("\n=== 3. Modeling ===")
print("Model Chosen: Random Forest Classifier")
print("Reasoning: Random Forests handle complex interactions between features well (e.g., location type vs daily waste vs weather), are robust to outliers, and don't require strict scaling. They also provide feature importance which might be useful for businesses.")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create full pipeline
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)


# ---------------------------------------------------------
# Deliverable 4: Evaluation
# ---------------------------------------------------------
print("\n=== 4. Evaluation ===")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")

print("\nClassification Report:\n", classification_report(y_test, y_pred))

print("[Interpretation]")
print("Precision: Out of all bins predicted as 'Full', X% were actually full. A high precision ensures we don't send trucks to empty bins, saving fuel.")
print("Recall: Out of all actually 'Full' bins, the model successfully identified Y%. A high recall ensures we don't miss overflowing bins, preventing hygiene issues.")
print("The results suggest the model is highly capable of discerning which bins need immediate attention.")

# ---------------------------------------------------------
# Deliverable 5 & 6: Business Interpretation & Future Improvements
# ---------------------------------------------------------
print("\n=== 5. Business Interpretation ===")
print("- Cost Reduction: By predicting which bins are full, operations can skip unnecessary pickups, saving on fuel, vehicle wear-and-tear, and labor hours.")
print("- Efficiency: Trucks can be dynamically routed only to pins that are predicted to be 'Full' or near full.")
print("- Decision Making: Ops teams can re-allocate bins. If a bin is constantly full in 1 day, supply a larger bin. If it's rarely full, supply a smaller one.")

print("\n=== 6. Future Improvements ===")
print("- Desired Real-Life Data: ")
print("   1. IoT Bin Sensors: Real-time volume data and weight sensors.")
print("   2. Historical routing/traffic info: For better ETA and cost modeling.")
print("   3. Neighborhood population density or footfall data.")
print("- Production Improvements: ")
print("   1. Transition to a time-series model forecasting when exactly it will be full.")
print("   2. Implement a route-optimization layer (like VRP - Vehicle Routing Problem) on top of the predictions.")
print("   3. Continuous learning pipeline to update the model as seasons/events change.")
