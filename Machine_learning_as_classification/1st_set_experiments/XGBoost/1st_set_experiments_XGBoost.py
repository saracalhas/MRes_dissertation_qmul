#!/usr/bin/env python
# coding: utf-8

# ### XGBoost tested during the 1st set of experiments. This was run via bash.

# In[1]:


#import libraries 
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import os


# In[2]:


# Load the imputed dataset
data = pd.read_csv('your_directory')


# In[3]:


data.head(5)


# ### Preparing the data for Machine Learning

# In[4]:


# Drop the identified columns from the dataset
# Note: may need to use data_stats instead of data_stats_imputed (data stats imputed has the categorical variables imputed)

#removed ethnicity and used ethnic_group instead
columnns_to_drop_2 = ['eid', 'APOE4', 'ethnicity','LTFU', 'dementia_date', 'dementia_prevalence', 'dementia_incidence', 'Diagnosis_since_baseline', 'Diagnosis_years_since_baseline', 'Years_Group']

data = data.drop(columns=columnns_to_drop_2)


# In[5]:


# Identify the column for the Y variable
target_column = 'dementia_diagnosis'

# Split the data into features (X) and target (Y)
X = data.drop(columns=[target_column])
Y = data[target_column]

# Verify the split
print(f"Features (X) shape: {X.shape}")
print(f"Target (Y) shape: {Y.shape}")


# In[6]:


# Check data types of all columns
data_types = X.dtypes

# Identify categorical variables
categorical_columns = data_types[data_types == 'object'].index.tolist()

print(f'Categorical columns: {categorical_columns}')


# #### convert to categorical variable

# In[7]:


# Identify numeric columns
numeric_columns = [col for col in X if col not in categorical_columns]

# Replace any infinity values (np.inf and -np.inf) with NaN. This ensures that the KNN imputer can process the data without errors.
X.replace([np.inf, -np.inf], np.nan, inplace=True)

# Convert the 'APOE4_alleles' column to categorical
X['APOE4_alleles'] = X['APOE4_alleles'].astype('category')

# Add 'APOE4_alleles' to the list of categorical columns if not already present
if 'APOE4_alleles' not in categorical_columns:
    categorical_columns.append('APOE4_alleles')


# In[8]:


# One-Hot Encode categorical variables using pd.get_dummies
X_encoded = pd.get_dummies(X, columns=categorical_columns)


# In[9]:


# Recalculate numeric columns after encoding
numeric_columns = [col for col in X_encoded.columns if col not in X.columns or col in numeric_columns]


# In[10]:


X_encoded.head(10)


# ### Split in train and test sets

# In[11]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, Y, test_size=0.3, random_state=42, stratify=Y)

# Standardize numerical columns after splitting
scaler = StandardScaler()
X_train[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
X_test[numeric_columns] = scaler.transform(X_test[numeric_columns])

# Verify the split
print(f"Training features (X_train) shape: {X_train.shape}")
print(f"Testing features (X_test) shape: {X_test.shape}")
print(f"Training target (y_train) shape: {y_train.shape}")
print(f"Testing target (y_test) shape: {y_test.shape}")


# In[12]:


# Convert to pandas Series if not already
y_train_series = pd.Series(y_train)
y_test_series = pd.Series(y_test)

# Get value counts for training and testing target
train_counts = y_train_series.value_counts()
test_counts = y_test_series.value_counts()

# Print the value counts
print("Training target distribution:")
print(train_counts)
print("\nTesting target distribution:")
print(test_counts)


# ### Experiment 1: simpler parameters to start and test XGBoost

# In[ ]:


import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, StratifiedShuffleSplit, train_test_split, cross_validate
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, make_scorer, confusion_matrix, roc_auc_score
import pickle

# Custom scoring function for specificity
def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

# Calculate scale_pos_weight
scale_pos_weight = float(sum(y_train == 0)) / sum(y_train == 1)

# Define a reduced parameter grid
param_dist = {
    'n_estimators': [100, 200],  # Smaller number of trees
    'max_depth': [3, 5, 7],  # Reduced depth
    'min_child_weight': [1, 5, 10],  # Reduced range
    'scale_pos_weight': [scale_pos_weight],  # Handling class imbalance
    'max_bin': [128],  # Fixed bin size
    'subsample': [0.7],  # Fixed subsample ratio
    'colsample_bytree': [0.3, 0.5],  # Reduced feature fraction
}

# XGBoost classifier with early stopping
xgb_model = xgb.XGBClassifier(tree_method='hist', device='cuda', use_label_encoder=False, random_state=42)


# Stratified shuffle split for faster cross-validation
inner_cv = StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=42)
outer_cv = StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=42)

# Define custom scoring functions
scoring = {
    'roc_auc': 'roc_auc',
    'recall': 'recall',
    'precision': 'precision',
    'f1': 'f1',
    'specificity': make_scorer(specificity_score)
}

# Randomized search
random_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_dist, cv=inner_cv, n_iter=10, scoring='roc_auc', n_jobs=2, verbose=2, random_state=42)

# Nested cross-validation
print("Starting nested cross-validation...")
nested_cv_results = cross_validate(random_search, X_train, y_train, cv=outer_cv, scoring=scoring, n_jobs=2, verbose=1, return_train_score=True)

# Print the mean and standard deviation of the nested cross-validation scores
print(f'Nested CV ROC AUC Model 1: {nested_cv_results["test_roc_auc"].mean():.4f} ± {nested_cv_results["test_roc_auc"].std():.4f}')
print(f'Nested CV Recall Model 1: {nested_cv_results["test_recall"].mean():.4f} ± {nested_cv_results["test_recall"].std():.4f}')
print(f'Nested CV Precision Model 1: {nested_cv_results["test_precision"].mean():.4f} ± {nested_cv_results["test_precision"].std():.4f}')
print(f'Nested CV F1 Model 1: {nested_cv_results["test_f1"].mean():.4f} ± {nested_cv_results["test_f1"].std():.4f}')
print(f'Nested CV Specificity Model 1: {nested_cv_results["test_specificity"].mean():.4f} ± {nested_cv_results["test_specificity"].std():.4f}')

# Fit the model
print("Fitting the best model...")
random_search.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_test, y_test)], verbose=10)

# Save the final model
def save_checkpoint(model, filename="checkpoint.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(model, f)
save_checkpoint(random_search, "final_model.pkl")

# Best hyperparameters
best_params = random_search.best_params_
print(f'Best hyperparameters Model 1: {best_params}')

# Test set evaluation
best_xgb = random_search.best_estimator_

y_pred_proba = best_xgb.predict_proba(X_test)[:, 1]
test_roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f'Test set ROC AUC Model 1: {test_roc_auc:.4f}')

# Additional test set metrics
y_pred = best_xgb.predict(X_test)
test_recall = recall_score(y_test, y_pred)
test_precision = precision_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred)
test_specificity = specificity_score(y_test, y_pred)

print(f'Test set Recall Model 1: {test_recall:.4f}')
print(f'Test set Precision Model 1: {test_precision:.4f}')
print(f'Test set F1 Model 1: {test_f1:.4f}')
print(f'Test set Specificity Model 1: {test_specificity:.4f}')


# In[ ]:


from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay, recall_score, precision_score, f1_score, roc_curve, roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt

# make predictions using the best model from cross validation in test dataset
y_pred= best_xgb.predict(X_test)

# Classification report
print("Classification Report Model 1:")
print(classification_report(y_test, y_pred))

# Create the output directory if it doesn't exist
output_dir = 'ML_work_v11_XG_Boost_no_feature_selection_only_shap'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
# Confusion matrix
print("Confusion Matrix Model 1:")
ConfusionMatrixDisplay.from_estimator(best_xgb, X_test, y_test, cmap=plt.cm.Blues)
plt.show()

# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'Confusion Matrix_model1.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Model 1: {accuracy:.2f}")

# Recall score (sensitivity)
recall = recall_score(y_test, y_pred)
print(f"Recall (Sensitivity) Model 1: {recall:.2f}")

# Specificity
specificity = recall_score(y_test, y_pred, pos_label=0)
print(f"Specificity Model 1: {specificity:.2f}")

# Precision
precision = precision_score(y_test, y_pred)
print(f"Precision Model 1: {precision:.2f}")

# F1 score
f1 = f1_score(y_test, y_pred)
print(f"F1 Score Model 1: {f1:.2f}")

# ROC AUC
y_pred_proba = best_xgb.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC AUC Model 1: {roc_auc:.2f}")

# Plot ROC Curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Model 1')
plt.legend(loc='lower right')
plt.show()

# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'Receiver Operating Characteristic_model1.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory

#this is a new item
# Precision-Recall Curve
precisions, recalls, _ = precision_recall_curve(y_test, y_pred)
plt.figure()
plt.plot(recalls, precisions, color='blue', lw=2, label=f'PR curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve Model 1')
plt.legend(loc='lower left')
plt.show()

# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'Precision-Recall Curve_model1.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory


# In[ ]:


import matplotlib.pyplot as plt

# Extract feature names from X_encoded
feature_names = X_encoded.columns

# Gain-based feature importance
importance_gain = best_xgb.get_booster().get_score(importance_type='gain')

# Convert to DataFrame and map feature names
importance_gain_df = pd.DataFrame(importance_gain.items(), columns=['Feature', 'Gain'])

def get_feature_name(feature):
    if feature.startswith('f') and feature[1:].isdigit():
        try:
            # Extract the index after 'f' and map to feature names
            index = int(feature[1:])
            return feature_names[index]
        except (IndexError, ValueError) as e:
            print(f"Error parsing feature name: {feature}, Error: {e}")
            return feature
    else:
        # Return the feature as is if it does not follow the 'f' format
        return feature

importance_gain_df['Feature'] = importance_gain_df['Feature'].apply(get_feature_name)

# Sort and display
importance_gain_df = importance_gain_df.sort_values(by='Gain', ascending=False)
print("Top features based on gain Model 1:")
print(importance_gain_df.head(10))

# Plot top features by gain
plt.figure(figsize=(10, 6))
plt.barh(importance_gain_df['Feature'].head(10), importance_gain_df['Gain'].head(10))
plt.xlabel('Gain')
plt.ylabel('Feature')
plt.title('Top 10 Features by Gain Model 1')
plt.gca().invert_yaxis()
plt.show()

# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'Top 10 Features by Gain_model1.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory



# ### Shap

# In[ ]:


import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Assuming best_xgb is your trained XGBoost model and X_encoded is the DataFrame before standardization

# Create a SHAP explainer for your model
explainer = shap.Explainer(best_xgb)

# Calculate SHAP values for the training set
shap_values = explainer(X_train)

# SHAP summary plot (bar)
shap.summary_plot(shap_values, features=X_train, feature_names=X_encoded.columns, plot_type="bar")

# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'SHAP summary plot_bar_model1.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory

# SHAP detailed summary plot
shap.summary_plot(shap_values, features=X_train, feature_names=X_encoded.columns)

# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'SHAP summary plot_model1.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory


# In[ ]:


# Create a SHAP explainer for your model
explainer = shap.Explainer(best_xgb)

# Calculate SHAP values for the test set
shap_values = explainer(X_test)

# SHAP summary plot (bar)
shap.summary_plot(shap_values, features=X_test, feature_names=X_encoded.columns, plot_type="bar")
output_path = os.path.join(output_dir, 'shap_values_bar_model1_test.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory

# SHAP detailed summary plot
shap.summary_plot(shap_values, features=X_test, feature_names=X_encoded.columns)
output_path = os.path.join(output_dir, 'shap_values_model1_test.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory

