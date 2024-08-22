#!/usr/bin/env python
# coding: utf-8

# ### This script includes the work for survival machine learning

# ### Preparing the data

# In[2]:


# Load the imputed dataset
data = pd.read_csv('path_to_file')


# In[3]:


data.info()


# In[4]:


data.head(5)


# In[5]:


#removed ethnicity and used ethnic_group instead
#removed APOE4 as per guidance and retained APOE4_alleles
# Load the datasets

columnns_to_drop_2 = ['APOE4', 'ethnicity','LTFU', 'dementia_date', 'dementia_prevalence', 'dementia_incidence', 'Diagnosis_since_baseline', 'Years_Group']

data = data.drop(columns=columnns_to_drop_2)


# In[6]:


# Print the names of the columns
print("Column names:", data.columns.tolist())


# In[7]:


# Identify the column for the Y variable
# Define the time and event columns for survival analysis
time_column = 'Diagnosis_years_since_baseline'
event_column = 'dementia_diagnosis'

columns_to_drop = ['eid', 'Diagnosis_years_since_baseline', 'dementia_diagnosis', time_column, event_column]

# Split data into features and target
X = data.drop(columns=columns_to_drop)
Y = data[[time_column, event_column]]


# In[8]:


# Verify the split
print(f"Features (X) shape: {X.shape}")
print(f"Target (Y) shape: {Y.shape}")


# In[9]:


Y.value_counts()


# In[10]:


# Identify categorical columns
categorical_columns = X.select_dtypes(include=['object']).columns.tolist()

print(f'Categorical columns: {categorical_columns}')


# In[11]:


# Convert the 'APOE4_alleles' column to categorical
X['APOE4_alleles'] = X['APOE4_alleles'].astype('category')

# Add 'APOE4_alleles' to the list of categorical columns if not already present
if 'APOE4_alleles' not in categorical_columns:
    categorical_columns.append('APOE4_alleles')


# In[12]:


# Identify numeric columns
numeric_columns = [col for col in X if col not in categorical_columns]

# Replace any infinity values (np.inf and -np.inf) with NaN. This ensures that the KNN imputer can process the data without errors.
X.replace([np.inf, -np.inf], np.nan, inplace=True)


# In[13]:


# One-Hot Encode categorical variables using pd.get_dummies
X_encoded = pd.get_dummies(X, columns=categorical_columns)


# In[14]:


# Recalculate numeric columns after encoding
numeric_columns = [col for col in X_encoded.columns if col not in X.columns or col in numeric_columns]


# In[15]:


X_encoded.head(10)


# In[16]:


# Print the names of the columns
print("Column names:", X_encoded.columns.tolist())


# ### Split the data in training and test before Boruta

# In[17]:


# Stratify based on event indicator
stratify_col = Y[event_column]


# In[18]:


# Split the data
random_state = 42
X_train, X_test, y_train, y_test = train_test_split(X_encoded, Y, test_size=0.3, random_state=random_state, stratify=stratify_col)

# Verify the split
print(f"Training features (X_train) shape: {X_train.shape}")
print(f"Testing features (X_test) shape: {X_test.shape}")
print(f"Training target (y_train) shape: {y_train.shape}")
print(f"Testing target (y_test) shape: {y_test.shape}")


# ### Feature selection with Boruta

# In[19]:


from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier

# Initialize a Random Forest model for Boruta
rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5, n_estimators=100, random_state=random_state)

# Initialize Boruta
boruta_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=random_state, max_iter=50)

# Run Boruta
boruta_selector.fit(X_train.values, y_train[event_column].values)

# Get the selected features
selected_features = X_train.columns[boruta_selector.support_].tolist()
print(f'Selected features: {selected_features}')


# In[20]:


# Select only the features identified by Boruta
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]
selected_feature_names = np.array(X_encoded.columns)[boruta_selector.support_]
print("Selected features:", selected_feature_names)
print("Number of selected features:", X_train_selected.shape[1])


# In[21]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier

# Fit the RandomForestClassifier again on the selected features to get feature importances
rf_selected = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=7, n_estimators=100, random_state=42)
rf_selected.fit(X_train_selected, y_train[event_column].values)
feature_importances = rf_selected.feature_importances_

# Create a DataFrame with feature names and their importances
importance_df = pd.DataFrame({
    'Feature': selected_features,
    'Importance': feature_importances
})

# Sort the DataFrame by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Create the output directory if it doesn't exist
output_dir = 'ML_work_v14_XGBoost_feature_reduction_with_apoe4alleles_only_boruta_Shapley_test'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Plot the top features
plt.figure(figsize=(10, 8))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Top Features Selected by Boruta')
plt.gca().invert_yaxis()
plt.show()

# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'Top Features Selected by Boruta.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory


# In[22]:


import matplotlib.pyplot as plt

# Plot the top features with improved visualization
plt.figure(figsize=(14, 30))  # Increase the height of the figure

# Create the bar plot with adjusted bar height
bars = plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue', edgecolor='black', height=0.5)

# Set the labels and title
plt.xlabel('Importance', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.title('Top Features Selected by Boruta', fontsize=16)
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Add data labels
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.001, bar.get_y() + bar.get_height() / 2, 
             f'{width:.3f}', ha='left', va='center', fontsize=10)

# Adjust font size for feature names and rotate y-axis labels if needed
plt.yticks(fontsize=10)
plt.xticks(fontsize=10)

# Adjust subplots to add more space on the left
plt.subplots_adjust(left=0.35, top=0.95, bottom=0.05)  # Increase the left margin

# Show and save the plot
plt.show()

# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'Top Features Selected by Boruta_3.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory

# Display the top features with their importances
importance_df


# ### Prepare the data for Survival ML work

# In[23]:


# Standardize the data after splitting
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_test_scaled = scaler.transform(X_test_selected)


# In[24]:


# Verify the split
print(f"Training features (X_train) shape: {X_train_scaled.shape}")
print(f"Testing features (X_test) shape: {X_test_scaled.shape}")
print(f"Training target (y_train) shape: {y_train.shape}")
print(f"Testing target (y_test) shape: {y_test.shape}")


# ### XGBoost classifier

# In[25]:


import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer
from lifelines.utils import concordance_index
import xgboost as xgb
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier


# In[29]:


# Custom scoring function for concordance index (C-index)
def c_index(y_true, y_pred):
    return concordance_index(y_true[:, 0], -y_pred, y_true[:, 1])

# Define a parameter grid
param_dist = {
    'max_depth': [4, 6, 8, 10],
    'min_child_weight': [10, 20, 30, 40, 50],
    'subsample': [0.5, 0.8],
    'colsample_bytree': [0.1, 0.2, 0.3, 0.5],
    'learning_rate': [0.0001, 0.001, 0.01, 0.1],
    'gamma': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 1],
    'alpha': [0, 0.01, 0.1, 0.5, 1, 2, 5, 10],  # L1 regularization
    'lambda': [0, 0.01, 0.1, 0.5, 1, 2, 5, 10],  # L2 regularization
}

# Custom wrapper for XGBRegressor to work with RandomizedSearchCV
class XGBSurvivalWrapper(xgb.XGBRegressor):
    def __init__(self, n_estimators=100, **kwargs):
        super().__init__(n_estimators=n_estimators, **kwargs)

    def fit(self, X, y, **kwargs):
        dtrain = xgb.DMatrix(X, label=y[:, 0], weight=y[:, 1])
        self._Booster = xgb.train(self.get_params(), dtrain, num_boost_round=self.n_estimators, **kwargs)
        return self

    def predict(self, X, **kwargs):
        dtest = xgb.DMatrix(X)
        return self._Booster.predict(dtest, **kwargs)

# Initialize the wrapper with the survival objective
xgb_model = XGBSurvivalWrapper(objective='survival:cox', tree_method='hist', random_state=42, n_estimators=1000)

# Randomized search
random_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_dist, cv=3, n_iter=10, scoring=make_scorer(c_index, greater_is_better=True), n_jobs=2, verbose=2, random_state=42)

# Nested cross-validation for survival analysis
scoring = {'c_index': make_scorer(c_index, greater_is_better=True)}

# Fit the model with nested cross-validation
nested_cv_results = cross_validate(random_search, X_train_scaled, y_train.to_numpy(), cv=3, scoring=scoring, n_jobs=2, verbose=1, return_train_score=True)

# Print the mean and standard deviation of the nested cross-validation scores
print(f'Nested CV C-index: {nested_cv_results["test_c_index"].mean():.4f} ± {nested_cv_results["test_c_index"].std():.4f}')

# Fit the final model
random_search.fit(X_train_scaled, y_train.to_numpy())

# Best hyperparameters
best_params = random_search.best_params_
print(f'Best hyperparameters: {best_params}')

# Test set evaluation
best_xgb = random_search.best_estimator_
y_pred = best_xgb.predict(X_test_scaled)

# Calculate and print the C-index for the test set
test_c_index = concordance_index(y_test[time_column], -y_pred, y_test[event_column])
print(f'Test set C-index: {test_c_index:.4f}')


# In[81]:


import shap
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt

# Initialize the SHAP explainer
explainer = shap.Explainer(best_xgb, X_test_scaled)

# Generate SHAP values for the test set
shap_values_test = explainer(X_test_scaled)

# Summary plot
shap.summary_plot(shap_values_test, X_test_scaled, feature_names=selected_feature_names)

# Bar plot
shap.summary_plot(shap_values_test, X_test_scaled, plot_type="bar", feature_names=selected_feature_names)


# In[ ]:




