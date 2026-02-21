import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


print("1. Data Exploration")
df = pd.read_csv('waste bin data.csv')
print(df.info())
print("\nMissing Values:\n", df.isnull().sum())
print("\nIs Full (Target):\n", df['is_full'].value_counts())


print("\n2. Feature Engineering")
df['avg_daily_waste_kg'] = df['avg_daily_waste_kg'].fillna(df['avg_daily_waste_kg'].median())
df['days_since_last_pickup'] = df['days_since_last_pickup'].fillna(df['days_since_last_pickup'].median())

df['estimated_current_waste_kg'] = df['avg_daily_waste_kg'] * df['days_since_last_pickup']
df['fill_ratio_estimate'] = df['estimated_current_waste_kg'] / df['bin_capacity_kg']
df['adjusted_fill_ratio'] = np.where(
    df['festival_week'] == 1,
    df['fill_ratio_estimate'] * 1.2,
    df['fill_ratio_estimate']
)

df = df.drop('bin_id', axis=1)

X = df.drop('is_full', axis=1)
y = df['is_full']

numerical_cols = ['avg_daily_waste_kg', 'days_since_last_pickup', 'festival_week',
                  'bin_capacity_kg', 'estimated_current_waste_kg',
                  'fill_ratio_estimate', 'adjusted_fill_ratio']
categorical_cols = ['location_type', 'weather']

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])


print("\n3. Training Random Forest with Hyperparameter Tuning")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

param_grid = {
    'classifier__n_estimators': [50, 100, 200, 300],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__max_features': ['sqrt', 'log2']
}

grid_search = GridSearchCV(
    rf_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"\nBest Parameters: {grid_search.best_params_}")
print(f"Best CV F1 Score: {grid_search.best_score_:.4f}")


print("\n4. Evaluation")
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))