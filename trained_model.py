import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load datasets
final_data = pd.read_csv("Data Sets/Final Data set.csv")
meal_suggestions = pd.read_csv("Data Sets/Meal suggestions.csv")
nutrients = pd.read_csv("Data Sets/Micro and macro nutrients.csv")

# Merge datasets on 'Daily_Calories'
merged_data = final_data.merge(meal_suggestions, on='Daily_Calories', how='left')
merged_data = merged_data.merge(nutrients, on='Daily_Calories', how='left')

# Encode categorical variables
label_encoders = {}
categorical_columns = ['Gender', 'Diet_Preference', 'Activity_Level', 'Disease', 'Food_Allergies', 'Health_Goal']
for col in categorical_columns:
    le = LabelEncoder()
    merged_data[col] = le.fit_transform(merged_data[col])
    label_encoders[col] = le

# Select features and target
X = merged_data.drop(columns=['Daily_Calories', 'Breakfast', 'Lunch', 'Dinner', 'Snacks'])
y = merged_data['Daily_Calories']

# Normalize numerical features
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")

# Save model
import joblib
joblib.dump(model, 'RandomForest1.pkl')
