import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


print("1. Data Exploration")
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



print("\n2. Feature Engineering")


df['avg_daily_waste_kg'] = df['avg_daily_waste_kg'].fillna(df['avg_daily_waste_kg'].median())
df['days_since_last_pickup'] = df['days_since_last_pickup'].fillna(df['days_since_last_pickup'].median())


df['estimated_current_waste_kg'] = df['avg_daily_waste_kg'] * df['days_since_last_pickup']

#
df['fill_ratio_estimate'] = df['estimated_current_waste_kg'] / df['bin_capacity_kg']


df['adjusted_fill_ratio'] = np.where(df['festival_week'] == 1, 
                                     df['fill_ratio_estimate'] * 1.2, 
                                     df['fill_ratio_estimate'])

print("Created features: 'estimated_current_waste_kg', 'fill_ratio_estimate', 'adjusted_fill_ratio'")
print("Reasoning: Understanding the accumulated waste over the days since last pickup relative to the bin capacity provides a direct proxy for the likelihood of the bin being full.")


df = df.drop('bin_id', axis=1)


X = df.drop('is_full', axis=1)
y = df['is_full']


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


print("\n3. Modeling ")
print("Model Chosen: Random Forest Classifier")
print("Reasoning: Random Forests handle complex interactions between features well (e.g., location type vs daily waste vs weather), are robust to outliers, and don't require strict scaling. They also provide feature importance which might be useful for businesses.")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Full pipeline
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])


model.fit(X_train, y_train)


y_pred = model.predict(X_test)



print("\n=== 4. Evaluation ===")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")

print("\nClassification Report:\n", classification_report(y_test, y_pred))

