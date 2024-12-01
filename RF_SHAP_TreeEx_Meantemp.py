# Import necessary libraries
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, YearLocator
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.tree import export_graphviz
from io import StringIO
from IPython.display import Image
import pydotplus


# Set random seeds
np.random.seed(42)

# Load and preprocess the dataset
filepath = 'D:\\pycharm\\pytorch_learn\\dataset\\archive\\\DailyDelhiClimate\\DailyDelhiClimateTrain_pre.csv'
data = pd.read_csv(filepath)
data = data.sort_values('date')
data['date'] = pd.to_datetime(data['date'])

# Extract and scale predictors
predictors = ['humidity','wind_speed','meanpressure']
scaler = MinMaxScaler(feature_range=(-1, 1))
data[predictors] = scaler.fit_transform(data[predictors])

# Split the dataset into historical (meantemp > 0) and future (meantemp = 0) parts
historical_data = data.loc[data['meantemp'] > 1]
future_data = data.loc[data['meantemp'] <= 1]
# Scale the target variable (meantemp) for historical data
historical_data['meantemp'] = scaler.fit_transform(historical_data['meantemp'].values.reshape(-1, 1))
correlation_matrix = historical_data.corr()
print(correlation_matrix['meantemp'].sort_values(ascending=False))

# Prepare training and validation sets from historical data
train_ratio = 0.8
train_end = int(train_ratio * len(historical_data))

X_train = historical_data[predictors][:train_end]
X_val = historical_data[predictors][train_end:]
X_test = future_data[predictors]
y_train = historical_data['meantemp'][:train_end]
y_val = historical_data['meantemp'][train_end:]
y_test = future_data['meantemp']  # These values are placeholders (0)

#以相关性为标准
# Use top features based on correlation
#important_features = correlation_matrix['meantemp'].sort_values(ascending=False).index[1:6].tolist()  # Top 5 features
#print(f"Selected Features: {important_features}")
# Prepare training and validation sets from historical data
#X_train = historical_data[important_features][:train_end]
#X_val = historical_data[important_features][train_end:]
#X_test = future_data[important_features]


# Initialize Random Forest model
rf_model = RandomForestRegressor(
    n_estimators=200,          # More trees for better stability
    max_depth=20,              # Limit depth to prevent overfitting
    max_features='sqrt',       # Use square root of features for splitting
    min_samples_split=5,       # Avoid overly deep trees
    min_samples_leaf=2,        # Ensure leaves have enough samples
    bootstrap=True,            # Default bootstrap for generalization
    oob_score=True,            # Use out-of-bag samples for validation
    n_jobs=-1,                 # Use all CPU cores for faster training
    random_state=42            # Reproducibility
)


# Train the model
rf_model.fit(X_train, y_train)
scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
print("Cross-Validation RMSE:", (-scores.mean()) ** 0.5)

# Evaluate the model
y_train_pred = rf_model.predict(X_train)
y_val_pred = rf_model.predict(X_val)
y_test_pred = rf_model.predict(X_test)  # Predict future meantemp
random_forest_predict = rf_model.predict(X_val)  # Predict future meantemp
random_forest_error=random_forest_predict-y_val


# Calculate metrics for training and validation sets
train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
val_rmse = mean_squared_error(y_val, y_val_pred, squared=False)
train_r2 = r2_score(y_train, y_train_pred)
val_r2 = r2_score(y_val, y_val_pred)

print(f"Train RMSE: {train_rmse}, R²: {train_r2}")
print(f"Validation RMSE: {val_rmse}, R²: {val_r2}")

# Combine predictions for plotting
future_years = future_data['date']
pred_y = np.concatenate([y_train_pred, y_val_pred, y_test_pred])
true_y = np.concatenate([y_train, y_val, [None] * len(y_test_pred)])  # No true values for future data
years = pd.concat([historical_data['date'], future_years])

# Plotting results
plt.figure(figsize=(15, 9))
plt.title("Random Forest Predictions vs. True Values")
x0 = [i for i in range(len(true_y))]
plt.plot(x0, pred_y, marker="o", markersize=1, label="Train Predictions")
plt.plot(x0, true_y, marker="o", markersize=1, label="Validation Predictions")
ax = plt.gca()  # Get current axis
ax.xaxis.set_major_locator(YearLocator(200))  # Major ticks every 2 years
plt.xlabel("date")
plt.ylabel("℃")
plt.legend()
plt.show()


# Draw test plot
# Draw decision tree visualizing plot
# Visualize one tree from the trained RandomForestRegressor
tree_index = 5  # meantempex of the tree to visualize
single_tree = rf_model.estimators_[tree_index]
# Export the tree structure to Graphviz format
dot_data = StringIO()
export_graphviz(single_tree,
                out_file=dot_data,
                feature_names=predictors,
                filled=True,
                rounded=True,
                special_characters=True)

# Convert the DOT file to a visual image
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png("tree_visualization.png")  # Save to file
Image(graph.create_png())  # Display the tree


# Feature importance
feature_importances = rf_model.feature_importances_
importance_df = pd.DataFrame({'Feature': predictors, 'Importance': feature_importances})
#importance_df = pd.DataFrame({'Feature': important_features, 'Importance': feature_importances}) #以相关性为标准
importance_df = importance_df.sort_values('Importance', ascending=False)

# Print feature importances
print("\nFeature Importances:")
print(importance_df)

# Plot feature importances
plt.figure(figsize=(12, 6))
sns.barplot(data=importance_df, x='Importance', y='Feature', palette="viridis")
plt.title('Feature Importances from Random Forest')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()


#SHAP model
import shap  # Add SHAP library for explainability
import IPython
from adjustText import adjust_text
# Calculate SHAP values using TreeExplainer
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_train[:100])
expected_value = explainer.expected_value
shap.initjs()

# Visualize feature contributions for the first validation instance
print("\nSHAP Feature Contributions for a Validation Instance:")
shap.force_plot(expected_value[0], shap_values[0], X_train.iloc[0,:], feature_names=predictors,matplotlib=True,show=True)

shap.plots.waterfall(shap.Explanation(values=shap_values[0],
                                      base_values=expected_value[0],
                                      feature_names=predictors))

shap.plots.bar(shap.Explanation(values=shap_values,
                                base_values=expected_value[0],
                                feature_names=predictors), show_data=True)
# Summary plot to visualize feature importance and contributions across the validation set
plt.title("SHAP Summary Plot")
shap.summary_plot(shap_values, X_train.iloc[:100,:],  feature_names=predictors,cmap="coolwarm")
shap.summary_plot(shap_values, X_train.iloc[:100,:],  feature_names=predictors, plot_type="bar")

# SHAP dependence plot for a specific feature (e.g., 'humidity')
plt.title("SHAP Dependence Plot for 'humidity'")
feature_index = predictors.index('wind_speed')  # Find the index of the feature
shap.dependence_plot(feature_index, shap_values[:100,:], X_train.iloc[:100,:], feature_names=predictors, interaction_index='meanpressure')
shap.dependence_plot('wind_speed', shap_values[:100,:], X_train.iloc[:100,:], feature_names=predictors, interaction_index=None)

# SHAP decision plot for a single prediction
shap.decision_plot(expected_value[0], shap_values[0], X_train)
shap.decision_plot(expected_value[0], shap_values[:100, :], X_train)
shap.decision_plot(expected_value[0], shap_values[:100, :], X_train, feature_order='hclust')
shap.decision_plot(expected_value[0], shap_values[:100, :] , X_train, link='logit')

# SHAP interaction plot
shap_interaction_values = explainer.shap_interaction_values(X_train.iloc[:100])
shap.summary_plot(
    shap_interaction_values[:100, :,1],  # Values for the first class (if multi-class, you can select the corresponding index)
    X_train.iloc[:100, :],  # The data used to calculate the SHAP values
    feature_names=predictors  # The feature names
)

# SHAP heatmap plot
shap.plots.heatmap(
    shap.Explanation(
        values=shap_values[:100, :],
        base_values=expected_value[0],
        data=X_train.iloc[:100, :],
        feature_names=predictors
    ),
    max_display=10,feature_values=shap.Explanation.abs.mean(0)) # Adjust the number of features displayed
