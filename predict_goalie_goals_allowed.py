import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Example: Load your data (update the path and columns as needed)
# The dataset should have columns like: ['games_played', 'shots_against', 'save_percentage', 'team_defense_rating', 'goals_allowed']
df = pd.read_csv('nhl_goalie_stats.csv')

# Features and target
features = ['games_played', 'shots_against', 'save_percentage', 'team_defense_rating']
target = 'goals_allowed'

# Prepare data
X = df[features]
y = df[target]

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
score = model.score(X_test, y_test)
print(f"Model R^2 Score: {score:.2f}")

# Predict function
def predict_goals_allowed(games_played, shots_against, save_percentage, team_defense_rating):
    features = np.array([[games_played, shots_against, save_percentage, team_defense_rating]])
    return model.predict(features)[0]

# Example usage
example_prediction = predict_goals_allowed(50, 1500, 0.915, 7.5)
print(f"Predicted Goals Allowed: {example_prediction:.1f}")

# To use your own data, replace the 'nhl_goalie_stats.csv' with your dataset.
# The features and model can be tuned for better accuracy depending on available data.