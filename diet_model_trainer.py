import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

# --- Load Diet Dataset (Dataset) ---
diet_df = pd.read_csv('diet_recommendations.csv')


print("Diet Dataset Loaded. Shape:", diet_df.shape)
print("Diet Dataset Columns:", diet_df.columns)



# Calculate BMI if not present or to ensure consistency
diet_df['Height_m'] = diet_df['Height_cm'] / 100
diet_df['BMI_calculated'] = diet_df['Weight_kg'] / (diet_df['Height_m'] ** 2)

# Select features we will use (based on what we plan to ask the user)
# Note: We are simplifying. A real system might need more complex mapping for Disease_Type
features_for_diet_model = [
    'Age', 'Gender', 'Weight_kg', 'Height_cm', 'BMI_calculated',
    'Disease_Type', 'Physical_Activity_Level',
    'Dietary_Restrictions', 'Allergies'
]
target_diet = 'Diet_Recommendation'

X = diet_df[features_for_diet_model].copy() # Use .copy() to avoid SettingWithCopyWarning
y = diet_df[target_diet]

# Handle missing values (using simple imputation)
# For categorical features, use most frequent. For numerical, use median.
# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns
numerical_cols = X.select_dtypes(include=['number']).columns

# Preprocessing pipelines
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()) # Scaling is good practice for many models
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # sparse_output=False for easier handling
])

# Create a preprocessor object using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ],
    remainder='passthrough' # Keep any columns not specified (should be none here)
)

# --- Train Diet Recommendation Model ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Full pipeline with preprocessor and classifier
diet_model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
])

print("Training Diet Model...")
diet_model_pipeline.fit(X_train, y_train)

# Evaluate (optional here, but good practice)
accuracy = diet_model_pipeline.score(X_test, y_test)
print(f"Diet Model Accuracy: {accuracy:.2f}")

# Save the trained model and the preprocessor's fitted columns for consistent input
model_path = 'diet_model.pkl'
joblib.dump(diet_model_pipeline, model_path)
print(f"Diet model saved to {model_path}")

# To ensure consistent one-hot encoding during prediction, we need to know the categories.
# One way is to save the fitted OneHotEncoder's categories or the preprocessor itself.
# The pipeline approach handles this internally, but if you were doing it manually:
# fitted_preprocessor = diet_model_pipeline.named_steps['preprocessor']
# joblib.dump(fitted_preprocessor, 'models/diet_preprocessor.pkl')
# print("Diet preprocessor saved.")
# For this simple project, saving the whole pipeline is sufficient.

print("Diet model training complete.")