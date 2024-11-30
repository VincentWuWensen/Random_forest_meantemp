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
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV

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
#important_features = correlation_matrix['Ind'].sort_values(ascending=False).index[1:6].tolist()  # Top 5 features
#print(f"Selected Features: {important_features}")
# Prepare training and validation sets from historical data
#X_train = historical_data[important_features][:train_end]
#X_val = historical_data[important_features][train_end:]
#X_test = future_data[important_features]


# Hyper-parametric randomized matching optimization (possibly locally optimal)
n_estimators_range=[int(x) for x in np.linspace(start=50,stop=1000,num=50)]
max_features_range=['auto','sqrt']
max_depth_range=[int(x) for x in np.linspace(10,500,num=50)]
max_depth_range.append(None)
min_samples_split_range=[2,5,10]
min_samples_leaf_range=[1,2,4,8]
bootstrap_range=[True,False]

random_forest_hp_range={'n_estimators':n_estimators_range,
                        'max_features':max_features_range,
                        'max_depth':max_depth_range,
                        'min_samples_split':min_samples_split_range,
                        'min_samples_leaf':min_samples_leaf_range
                        # 'bootstrap':bootstrap_range
                        }
pprint(random_forest_hp_range)
random_forest_model_test_base=RandomForestRegressor()
rf_model = RandomizedSearchCV(estimator=random_forest_model_test_base,
                                                   param_distributions=random_forest_hp_range,
                                                   n_iter=200,
                                                   n_jobs=-1,
                                                   cv=3,
                                                   verbose=1,
                                                   random_state=42
                                                   )
rf_model.fit(X_train, y_train)
best_hp_now=rf_model.best_params_
pprint(best_hp_now)


#Hyper-parametric randomized matching optimization
param_grid = {
    'n_estimators': [350, 418, 500],
    'max_features': [2, 3],
    'max_depth': [8, 10, 15],
    'min_samples_split': [1, 2],  # Greater than 1
    'min_samples_leaf': [1, 2]
    # 'bootstrap':bootstrap_range
}
grid_search = GridSearchCV(RandomForestRegressor(random_state=42),
                           param_grid, cv=3, verbose=1, n_jobs=-1, scoring='neg_mean_squared_error', return_train_score=True)

# Print detailed results of hyperparameter tuning
grid_search.fit(X_train, y_train)
print("\nGrid Search Results:")
results = pd.DataFrame(grid_search.cv_results_)
print(results[['params', 'mean_test_score', 'mean_train_score']])
print("\nBest Parameters:", grid_search.best_params_)
best_hp_now_2=grid_search.best_params_
pprint(best_hp_now_2)


# Train the optimized model
rf_model = grid_search.best_estimator_

# Train the model
rf_model.fit(X_train, y_train)
scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
print("Cross-Validation RMSE:", (-scores.mean()) ** 0.5)

# Evaluate the model
y_train_pred = rf_model.predict(X_train)
y_val_pred = rf_model.predict(X_val)
y_test_pred = rf_model.predict(X_test)  # Predict future `Ind`

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
tree_index = 5  # Index of the tree to visualize
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

# Re-standardizing with random forests
feature_importances = rf_model.feature_importances_
important_features = [predictors[i] for i in np.argsort(feature_importances)[::-1][:5]]  # Top 5 features
X_train = historical_data[important_features][:train_end]
X_val = historical_data[important_features][train_end:]
X_test = future_data[important_features]
#Return up to reforecast