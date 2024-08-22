#!/usr/bin/env python
# coding: utf-8

# ### This notebook contains experiment 1 to 11 of the XGBoost experiments. These include all variables. This code was run via bash.

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


# Load the imputed dataset after KNN imputation
data = pd.read_csv('your_directory_to_file')


# In[3]:


data.head(5)


# ### Preparing the data for Machine Learning

# In[4]:


# Drop the identified columns from the dataset

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


# ### XGBoost classifiers

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


# ### Lime

# In[ ]:


import os
import numpy as np
import pandas as pd
import lime
import lime.lime_tabular
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from IPython.display import display, Image

# Predict on the test set
y_pred = best_xgb.predict(X_test)
y_pred_proba = best_xgb.predict_proba(X_test)

# Separate positive and negative examples with confidence > 80%
positive_indices = np.where((y_pred == 1) & (y_pred_proba[:, 1] > 0.8))[0]
negative_indices = np.where((y_pred == 0) & (y_pred_proba[:, 0] > 0.8))[0]

# Ensure there are examples to choose from
if len(positive_indices) == 0:
    print("No positive examples with confidence > 80% found.")
if len(negative_indices) == 0:
    print("No negative examples with confidence > 80% found.")

# Select multiple instances to explain
# Example: Select the first two positive and negative examples
num_examples = 2
positive_instance_indices = positive_indices[:num_examples] if len(positive_indices) >= num_examples else positive_indices
negative_instance_indices = negative_indices[:num_examples] if len(negative_indices) >= num_examples else negative_indices

# Initialize LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train.values,
    feature_names=X_encoded.columns,
    class_names=np.unique(Y).astype(str),
    discretize_continuous=True
)

# Function to display and save explanations
def display_and_save_explanations(instance_indices, label):
    for idx in instance_indices:
        instance = X_test.iloc[idx].values  # Access the instance using iloc for positional indexing
        confidence = y_pred_proba[idx][1] if y_pred[idx] == 1 else y_pred_proba[idx][0]
        print(f"Confidence for the {label} instance {idx}: {confidence:.2f}")
        exp = explainer.explain_instance(instance, best_xgb.predict_proba, num_features=10)
        
        # Display in the notebook
        exp.show_in_notebook(show_table=True, show_all=False)
        
        # Save the explanation as an image
        fig = exp.as_pyplot_figure()
        plt.tight_layout()
        image_path = os.path.join(output_dir, f'lime_explanation_{label}_best_xgb_1_{idx}.png')
        fig.savefig(image_path, format='png', bbox_inches='tight')
        plt.close(fig)
        print(f'Saved explanation to {image_path}')
        
        # Display the saved image in the notebook
        display(Image(filename=image_path))

# Display and save explanations for positive instances
if len(positive_instance_indices) > 0:
    print("Positive Instances with Confidence > 80%:")
    display_and_save_explanations(positive_instance_indices, 'positive')

# Display and save explanations for negative instances
if len(negative_instance_indices) > 0:
    print("Negative Instances with Confidence > 80%:")
    display_and_save_explanations(negative_instance_indices, 'negative')


# ### Experiment 2: larger number of n_estimators & different learning rates

# In[ ]:


import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, StratifiedShuffleSplit, train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, make_scorer, confusion_matrix, accuracy_score
import pickle

# Custom specificity score function
def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

# Calculate scale_pos_weight
scale_pos_weight = float(sum(y_train == 0)) / sum(y_train == 1)

# Define a reduced parameter grid
param_dist = {
    'n_estimators': [1000],
    'max_depth': [4, 6, 8, 10],
    'min_child_weight': [10, 20, 30, 40, 50],
    'scale_pos_weight': [scale_pos_weight],
    'subsample': [0.5, 0.8],
    'colsample_bytree': [0.1, 0.2, 0.3, 0.5],
    'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.3],
}

# XGBoost classifier with early stopping
xgb_model= xgb.XGBClassifier(tree_method='hist', device='cuda', use_label_encoder=False, random_state=42)

# Stratified shuffle split for faster cross-validation
inner_cv = StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=42)
outer_cv = StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=42)

# Define custom scoring functions
scoring = {
    'roc_auc': 'roc_auc',
    'recall': 'recall',
    'precision': 'precision',
    'f1': 'f1',
    'specificity': make_scorer(specificity_score),
    'accuracy': 'accuracy'
}

# Randomized search for cross validation
random_search_2 = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_dist, cv=inner_cv, n_iter=10, scoring='roc_auc', n_jobs=2, verbose=2, random_state=42)

# Nested cross-validation
print("Starting nested cross-validation...")
nested_cv_results= cross_validate(random_search_2, X_train, y_train, cv=outer_cv, scoring=scoring, n_jobs=2, verbose=1, return_train_score=True)

# Print the mean and standard deviation of the nested cross-validation scores
print(f'Nested CV ROC AUC Model 2: {nested_cv_results["test_roc_auc"].mean():.4f} ± {nested_cv_results["test_roc_auc"].std():.4f}')
print(f'Nested CV Recall Model 2: {nested_cv_results["test_recall"].mean():.4f} ± {nested_cv_results["test_recall"].std():.4f}')
print(f'Nested CV Precision Model 2: {nested_cv_results["test_precision"].mean():.4f} ± {nested_cv_results["test_precision"].std():.4f}')
print(f'Nested CV F1 Model 2: {nested_cv_results["test_f1"].mean():.4f} ± {nested_cv_results["test_f1"].std():.4f}')
print(f'Nested CV Specificity Model 2: {nested_cv_results["test_specificity"].mean():.4f} ± {nested_cv_results["test_specificity"].std():.4f}')
print(f'Nested CV Accuracy Model 2: {nested_cv_results["test_accuracy"].mean():.4f} ± {nested_cv_results["test_accuracy"].std():.4f}')

# Fit the model
print("Fitting the best model...")
random_search_2.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_test, y_test)], verbose=10)

# Save the final model
def save_checkpoint(model, filename="checkpoint.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(model, f)
save_checkpoint(random_search_2, "final_model.pkl")

# Best hyperparameters
best_params_2 = random_search_2.best_params_
print(f'Best hyperparameters Model 2: {best_params_2}')

# Test set evaluation
best_xgb_2 = random_search_2.best_estimator_

y_pred_proba = best_xgb_2.predict_proba(X_test)[:, 1]
test_roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f'Test set ROC AUC Model 2: {test_roc_auc:.4f}')

# Additional test set metrics
y_pred = best_xgb_2.predict(X_test)
test_recall = recall_score(y_test, y_pred)
test_precision = precision_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred)
test_specificity = specificity_score(y_test, y_pred)

print(f'Test set Recall Model 2: {test_recall:.4f}')
print(f'Test set Precision Model 2: {test_precision:.4f}')
print(f'Test set F1 Model 2: {test_f1:.4f}')
print(f'Test set Specificity Model 2: {test_specificity:.4f}')


# In[ ]:


from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay, recall_score, precision_score, f1_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# make predictions using the best model from cross validation in test dataset
y_pred= best_xgb_2.predict(X_test)

# Classification report
print("Classification Report Model 2:")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("Confusion Matrix Model 2:")
ConfusionMatrixDisplay.from_estimator(best_xgb_2, X_test, y_test, cmap=plt.cm.Blues)
plt.show()

# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'Confusion Matrix_model2.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Model 2: {accuracy:.2f}")

# Recall score (sensitivity)
recall = recall_score(y_test, y_pred)
print(f"Recall (Sensitivity) Model 2: {recall:.2f}")

# Specificity
specificity = recall_score(y_test, y_pred, pos_label=0)
print(f"Specificity Model 2: {specificity:.2f}")

# Precision
precision = precision_score(y_test, y_pred)
print(f"Precision Model 2: {precision:.2f}")

# F1 score
f1 = f1_score(y_test, y_pred)
print(f"F1 Score Model 2: {f1:.2f}")

# ROC AUC
y_pred_proba = best_xgb_2.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC AUC: {roc_auc:.2f}")

# Plot ROC Curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Model 2')
plt.legend(loc='lower right')
plt.show()

# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'Receiver Operating Characteristic_model2.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory

#this is a new item
# Precision-Recall Curve
precisions, recalls, _ = precision_recall_curve(y_test, y_pred)
plt.figure()
plt.plot(recalls, precisions, color='blue', lw=2, label=f'PR curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve Model 2')
plt.legend(loc='lower left')
plt.show()

# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'Precision-Recall Curve_model2.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory


# In[ ]:


import matplotlib.pyplot as plt

# Extract feature names from X_encoded
feature_names = X_encoded.columns

# Gain-based feature importance
importance_gain = best_xgb_2.get_booster().get_score(importance_type='gain')

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
print("Top features based on gain Model 2:")
print(importance_gain_df.head(10))

# Plot top features by gain
plt.figure(figsize=(10, 6))
plt.barh(importance_gain_df['Feature'].head(10), importance_gain_df['Gain'].head(10))
plt.xlabel('Gain')
plt.ylabel('Feature')
plt.title('Top 10 Features by Gain Model 2')
plt.gca().invert_yaxis()
plt.show()

# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'Top 10 Features by Gain_model2.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory


# ### Shap

# In[ ]:


import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create a SHAP explainer for your model
explainer = shap.Explainer(best_xgb_2)

# Calculate SHAP values for the training set
shap_values = explainer(X_train)

# SHAP summary plot (bar)
shap.summary_plot(shap_values, features=X_train, feature_names=X_encoded.columns, plot_type="bar")

# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'SHAP summary plot_bar_model2.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory

# SHAP detailed summary plot
shap.summary_plot(shap_values, features=X_train, feature_names=X_encoded.columns)

# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'SHAP summary plot_model2.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory


# In[ ]:


# Create a SHAP explainer for your model
explainer = shap.Explainer(best_xgb_2)

# Calculate SHAP values for the test set
shap_values = explainer(X_test)

# SHAP summary plot (bar)
shap.summary_plot(shap_values, features=X_test, feature_names=X_encoded.columns, plot_type="bar")
output_path = os.path.join(output_dir, 'shap_values_bar_model2_test.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory

# SHAP detailed summary plot
shap.summary_plot(shap_values, features=X_test, feature_names=X_encoded.columns)
output_path = os.path.join(output_dir, 'shap_values_model2_test.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory


# ### Lime

# In[ ]:


import os
import numpy as np
import pandas as pd
import lime
import lime.lime_tabular
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from IPython.display import display, Image

# Predict on the test set
y_pred = best_xgb_2.predict(X_test)
y_pred_proba = best_xgb_2.predict_proba(X_test)

# Separate positive and negative examples with confidence > 80%
positive_indices = np.where((y_pred == 1) & (y_pred_proba[:, 1] > 0.8))[0]
negative_indices = np.where((y_pred == 0) & (y_pred_proba[:, 0] > 0.8))[0]

# Ensure there are examples to choose from
if len(positive_indices) == 0:
    print("No positive examples with confidence > 80% found.")
if len(negative_indices) == 0:
    print("No negative examples with confidence > 80% found.")

# Select multiple instances to explain
# Example: Select the first two positive and negative examples
num_examples = 2
positive_instance_indices = positive_indices[:num_examples] if len(positive_indices) >= num_examples else positive_indices
negative_instance_indices = negative_indices[:num_examples] if len(negative_indices) >= num_examples else negative_indices

# Initialize LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train.values,
    feature_names=X_encoded.columns,
    class_names=np.unique(Y).astype(str),
    discretize_continuous=True
)

# Function to display and save explanations
def display_and_save_explanations(instance_indices, label):
    for idx in instance_indices:
        instance = X_test.iloc[idx].values  # Access the instance using iloc for positional indexing
        confidence = y_pred_proba[idx][1] if y_pred[idx] == 1 else y_pred_proba[idx][0]
        print(f"Confidence for the {label} instance {idx}: {confidence:.2f}")
        exp = explainer.explain_instance(instance, best_xgb_2.predict_proba, num_features=10)
        
        # Display in the notebook
        exp.show_in_notebook(show_table=True, show_all=False)
        
        # Save the explanation as an image
        fig = exp.as_pyplot_figure()
        plt.tight_layout()
        image_path = os.path.join(output_dir, f'lime_explanation_{label}_best_xgb_2_{idx}.png')
        fig.savefig(image_path, format='png', bbox_inches='tight')
        plt.close(fig)
        print(f'Saved explanation to {image_path}')
        
        # Display the saved image in the notebook
        display(Image(filename=image_path))

# Display and save explanations for positive instances
if len(positive_instance_indices) > 0:
    print("Positive Instances with Confidence > 80%:")
    display_and_save_explanations(positive_instance_indices, 'positive')

# Display and save explanations for negative instances
if len(negative_instance_indices) > 0:
    print("Negative Instances with Confidence > 80%:")
    display_and_save_explanations(negative_instance_indices, 'negative')


# ### Experiment 3. Similar settings to exp 2 but with additional trees

# In[13]:


import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, StratifiedShuffleSplit, train_test_split, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, make_scorer, confusion_matrix, accuracy_score
import pickle

# Custom specificity score function
def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

# Calculate scale_pos_weight
scale_pos_weight = float(sum(y_train == 0)) / sum(y_train == 1)

# Define a reduced parameter grid
param_dist = {
    'n_estimators': [100, 200, 300, 500, 1000],
    'max_depth': [4, 6, 8, 10],
    'min_child_weight': [10, 20, 30, 40, 50],
    'scale_pos_weight': [scale_pos_weight],
    'subsample': [0.5, 0.8],
    'colsample_bytree': [0.1, 0.2, 0.3, 0.5],
    'learning_rate': [0.0001, 0.001, 0.01, 0.1],
}

# XGBoost classifier with GPU support
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
    'specificity': make_scorer(specificity_score),
    'accuracy': 'accuracy'
}

# Randomized search for cross validation
random_search_3 = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_dist, cv=inner_cv, n_iter=10, scoring='roc_auc', n_jobs=2, verbose=2, random_state=42)

# Nested cross-validation
print("Starting nested cross-validation...")
nested_cv_results = cross_validate(random_search_3, X_train, y_train, cv=outer_cv, scoring=scoring, n_jobs=2, verbose=1, return_train_score=True)

# Print the mean and standard deviation of the nested cross-validation scores
print(f'Nested CV ROC AUC Model 3: {nested_cv_results["test_roc_auc"].mean():.4f} ± {nested_cv_results["test_roc_auc"].std():.4f}')
print(f'Nested CV Recall Model 3: {nested_cv_results["test_recall"].mean():.4f} ± {nested_cv_results["test_recall"].std():.4f}')
print(f'Nested CV Precision Model 3: {nested_cv_results["test_precision"].mean():.4f} ± {nested_cv_results["test_precision"].std():.4f}')
print(f'Nested CV F1 Model 3: {nested_cv_results["test_f1"].mean():.4f} ± {nested_cv_results["test_f1"].std():.4f}')
print(f'Nested CV Specificity Model 3: {nested_cv_results["test_specificity"].mean():.4f} ± {nested_cv_results["test_specificity"].std():.4f}')
print(f'Nested CV Accuracy Model 3: {nested_cv_results["test_accuracy"].mean():.4f} ± {nested_cv_results["test_accuracy"].std():.4f}')

# Fit the model
print("Fitting the best model...")
random_search_3.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_test, y_test)], verbose=10)

# Save the final model
def save_checkpoint(model, filename="checkpoint.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(model, f)
save_checkpoint(random_search_3, "final_model.pkl")

# Best hyperparameters
best_params_3 = random_search_3.best_params_
print(f'Best hyperparameters Model 3: {best_params_3}')

# Test set evaluation
best_xgb_3 = random_search_3.best_estimator_

y_pred_proba = best_xgb_3.predict_proba(X_test)[:, 1]
test_roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f'Test set ROC AUC Model 3: {test_roc_auc:.4f}')

# Additional test set metrics
y_pred = best_xgb_3.predict(X_test)
test_recall = recall_score(y_test, y_pred)
test_precision = precision_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred)
test_specificity = specificity_score(y_test, y_pred)

print(f'Test set Recall Model 3: {test_recall:.4f}')
print(f'Test set Precision Model 3: {test_precision:.4f}')
print(f'Test set F1 Model 3: {test_f1:.4f}')
print(f'Test set Specificity Model 3: {test_specificity:.4f}')


# In[14]:


from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay, recall_score, precision_score, f1_score, roc_curve, roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt

# make predictions using the best model from cross validation in test dataset
y_pred= best_xgb_3.predict(X_test)

# Classification report
print("Classification Report Model 3:")
print(classification_report(y_test, y_pred))

# Create the output directory if it doesn't exist
output_dir = 'ML_work_v11_XG_Boost_no_feature_selection_only_shap'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Confusion matrix
print("Confusion Matrix Model 3:")
ConfusionMatrixDisplay.from_estimator(best_xgb_3, X_test, y_test, cmap=plt.cm.Blues)
plt.show()

# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'confusion_matrix_model3.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Model 3: {accuracy:.2f}")

# Recall score (sensitivity)
recall = recall_score(y_test, y_pred)
print(f"Recall (Sensitivity) Model 3: {recall:.2f}")

# Specificity
specificity = recall_score(y_test, y_pred, pos_label=0)
print(f"Specificity Model 3: {specificity:.2f}")

# Precision
precision = precision_score(y_test, y_pred)
print(f"Precision Model 3: {precision:.2f}")

# F1 score
f1 = f1_score(y_test, y_pred)
print(f"F1 Score Model 3: {f1:.2f}")

# ROC AUC
y_pred_proba = best_xgb_3.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC AUC Model 3: {roc_auc:.2f}")

# Plot ROC Curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Model 3')
plt.legend(loc='lower right')
plt.show()

# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'Receiver Operating Characteristic_model3.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory

#this is a new item
# Precision-Recall Curve
precisions, recalls, _ = precision_recall_curve(y_test, y_pred)
plt.figure()
plt.plot(recalls, precisions, color='blue', lw=2, label=f'PR curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve Model 3')
plt.legend(loc='lower left')
plt.show()

# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'Precision-Recall Curve_model3.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory


# In[15]:


import matplotlib.pyplot as plt

# Extract feature names from X_encoded
feature_names = X_encoded.columns

# Gain-based feature importance
importance_gain = best_xgb_3.get_booster().get_score(importance_type='gain')

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
print("Top features based on gain Model 3:")
print(importance_gain_df.head(10))

# Plot top features by gain
plt.figure(figsize=(10, 6))
plt.barh(importance_gain_df['Feature'].head(10), importance_gain_df['Gain'].head(10))
plt.xlabel('Gain')
plt.ylabel('Feature')
plt.title('Top 10 Features by Gain Model 3')
plt.gca().invert_yaxis()
plt.show()

# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'Top 10 Features by Gain_model3.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory


# In[16]:


import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create a SHAP explainer for your model
explainer = shap.Explainer(best_xgb_3)

# Calculate SHAP values for the training set
shap_values = explainer(X_train)

# SHAP summary plot (bar)
shap.summary_plot(shap_values, features=X_train, feature_names=X_encoded.columns, plot_type="bar")
# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'shap_values_bar_model3_train.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory


# SHAP detailed summary plot
shap.summary_plot(shap_values, features=X_train, feature_names=X_encoded.columns)
# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'shap_values_model3_train.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory


# In[17]:


# Create a SHAP explainer for your model
explainer = shap.Explainer(best_xgb_3)

# Calculate SHAP values for the test set
shap_values = explainer(X_test)

# SHAP summary plot (bar)
shap.summary_plot(shap_values, features=X_test, feature_names=X_encoded.columns, plot_type="bar")
output_path = os.path.join(output_dir, 'shap_values_bar_model3_test.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory

# SHAP detailed summary plot
shap.summary_plot(shap_values, features=X_test, feature_names=X_encoded.columns)
output_path = os.path.join(output_dir, 'shap_values_model3_test.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory


# ### LIME

# In[20]:


import os
import numpy as np
import pandas as pd
import lime
import lime.lime_tabular
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from IPython.display import display, Image

# Predict on the test set
y_pred = best_xgb_3.predict(X_test)
y_pred_proba = best_xgb_3.predict_proba(X_test)

# Separate positive and negative examples with confidence > 80%
positive_indices = np.where((y_pred == 1) & (y_pred_proba[:, 1] > 0.8))[0]
negative_indices = np.where((y_pred == 0) & (y_pred_proba[:, 0] > 0.8))[0]

# Ensure there are examples to choose from
if len(positive_indices) == 0:
    print("No positive examples with confidence > 80% found.")
if len(negative_indices) == 0:
    print("No negative examples with confidence > 80% found.")

# Select multiple instances to explain
# Example: Select the first two positive and negative examples
num_examples = 2
positive_instance_indices = positive_indices[:num_examples] if len(positive_indices) >= num_examples else positive_indices
negative_instance_indices = negative_indices[:num_examples] if len(negative_indices) >= num_examples else negative_indices

# Initialize LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train.values,
    feature_names=X_encoded.columns,
    class_names=np.unique(Y).astype(str),
    discretize_continuous=True
)

# Function to display and save explanations
def display_and_save_explanations(instance_indices, label):
    for idx in instance_indices:
        instance = X_test.iloc[idx].values  # Access the instance using iloc for positional indexing
        confidence = y_pred_proba[idx][1] if y_pred[idx] == 1 else y_pred_proba[idx][0]
        print(f"Confidence for the {label} instance {idx}: {confidence:.2f}")
        exp = explainer.explain_instance(instance, best_xgb_3.predict_proba, num_features=10)
        
        # Display in the notebook
        exp.show_in_notebook(show_table=True, show_all=False)
        
        # Save the explanation as an image
        fig = exp.as_pyplot_figure()
        plt.tight_layout()
        image_path = os.path.join(output_dir, f'lime_explanation_{label}_best_xgb_3_{idx}.png')
        fig.savefig(image_path, format='png', bbox_inches='tight')
        plt.close(fig)
        print(f'Saved explanation to {image_path}')
        
        # Display the saved image in the notebook
        display(Image(filename=image_path))

# Display and save explanations for positive instances
if len(positive_instance_indices) > 0:
    print("Positive Instances with Confidence > 80%:")
    display_and_save_explanations(positive_instance_indices, 'positive')

# Display and save explanations for negative instances
if len(negative_instance_indices) > 0:
    print("Negative Instances with Confidence > 80%:")
    display_and_save_explanations(negative_instance_indices, 'negative')


# ### Experiment 4: Like experiment 3 + gamma + stratified KFold

# In[ ]:


import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, make_scorer, confusion_matrix, accuracy_score
import pickle

# Custom scoring function for specificity
def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

# Calculate scale_pos_weight
scale_pos_weight = float(sum(y_train == 0)) / sum(y_train == 1)

# Define a parameter grid
param_dist = {
    'n_estimators': [100, 200, 300, 500, 1000],
    'max_depth': [4, 6, 8, 10],
    'min_child_weight': [10, 20, 30, 40, 50],
    'scale_pos_weight': [scale_pos_weight],
    'subsample': [0.5, 0.8],
    'colsample_bytree': [0.1, 0.2, 0.3, 0.5],
    'learning_rate': [0.0001, 0.001, 0.01, 0.1],
    'gamma': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 1],
}

# XGBoost classifier with GPU support
xgb_model = xgb.XGBClassifier(tree_method='hist', device='cuda', use_label_encoder=False, random_state=42)

# Stratified K-Fold cross-validation
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
outer_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Define custom scoring functions
scoring = {
    'roc_auc': 'roc_auc',
    'recall': 'recall',
    'precision': 'precision',
    'f1': 'f1',
    'specificity': make_scorer(specificity_score),
    'accuracy': 'accuracy'
}

# Randomized search
random_search_4 = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_dist, cv=inner_cv, n_iter=10, scoring='roc_auc', n_jobs=2, verbose=2, random_state=42)

# Nested cross-validation
nested_cv_results = cross_validate(random_search_4, X_train, y_train, cv=outer_cv, scoring=scoring, n_jobs=2, verbose=1, return_train_score=True)

# Print the mean and standard deviation of the nested cross-validation scores
print(f'Nested CV ROC AUC Model 4: {nested_cv_results["test_roc_auc"].mean():.4f} ± {nested_cv_results["test_roc_auc"].std():.4f}')
print(f'Nested CV Recall Model 4: {nested_cv_results["test_recall"].mean():.4f} ± {nested_cv_results["test_recall"].std():.4f}')
print(f'Nested CV Precision Model 4: {nested_cv_results["test_precision"].mean():.4f} ± {nested_cv_results["test_precision"].std():.4f}')
print(f'Nested CV F1 Model 4: {nested_cv_results["test_f1"].mean():.4f} ± {nested_cv_results["test_f1"].std():.4f}')
print(f'Nested CV Specificity Model 4: {nested_cv_results["test_specificity"].mean():.4f} ± {nested_cv_results["test_specificity"].std():.4f}')
print(f'Nested CV Accuracy Model 4: {nested_cv_results["test_accuracy"].mean():.4f} ± {nested_cv_results["test_accuracy"].std():.4f}')

# Fit the model
random_search_4.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_test, y_test)], verbose=10)

# Save the final model
def save_checkpoint(model, filename="checkpoint.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(model, f)
save_checkpoint(random_search_4, "final_model.pkl")

# Best hyperparameters
best_params_4 = random_search_4.best_params_
print(f'Best hyperparameters Model 4: {best_params_4}')

# Test set evaluation
best_xgb_4 = random_search_4.best_estimator_

y_pred_proba = best_xgb_4.predict_proba(X_test)[:, 1]
test_roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f'Test set ROC AUC Model 4: {test_roc_auc:.4f}')

# Additional test set metrics
y_pred = best_xgb_4.predict(X_test)
test_recall = recall_score(y_test, y_pred)
test_precision = precision_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred)
test_specificity = specificity_score(y_test, y_pred)

print(f'Test set Recall Model 4: {test_recall:.4f}')
print(f'Test set Precision Model 4: {test_precision:.4f}')
print(f'Test set F1 Model 4: {test_f1:.4f}')
print(f'Test set Specificity Model 4: {test_specificity:.4f}')


# In[ ]:


from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay, recall_score, precision_score, f1_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# make predictions using the best model from cross validation in test dataset
y_pred= best_xgb_4.predict(X_test)

# Classification report
print("Classification Report Model 4:")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("Confusion Matrix Model 4:")
ConfusionMatrixDisplay.from_estimator(best_xgb_4, X_test, y_test, cmap=plt.cm.Blues)
plt.show()

# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'Confusion Matrix_model4.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory


# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Model 4: {accuracy:.2f}")

# Recall score (sensitivity)
recall = recall_score(y_test, y_pred)
print(f"Recall (Sensitivity) Model 4: {recall:.2f}")

# Specificity
specificity = recall_score(y_test, y_pred, pos_label=0)
print(f"Specificity Model 4: {specificity:.2f}")

# Precision
precision = precision_score(y_test, y_pred)
print(f"Precision Model 4: {precision:.2f}")

# F1 score
f1 = f1_score(y_test, y_pred)
print(f"F1 Score Model 4: {f1:.2f}")

# ROC AUC
y_pred_proba = best_xgb_4.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC AUC Model 4: {roc_auc:.2f}")

# Plot ROC Curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Model 4')
plt.legend(loc='lower right')
plt.show()

# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'Receiver Operating Characteristic_model4.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory



#this is a new item
# Precision-Recall Curve
precisions, recalls, _ = precision_recall_curve(y_test, y_pred)
plt.figure()
plt.plot(recalls, precisions, color='blue', lw=2, label=f'PR curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve Model 4')
plt.legend(loc='lower left')
plt.show()

# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'Precision-Recall Curve_model4.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory


# In[ ]:


import matplotlib.pyplot as plt

# Extract feature names from X_encoded
feature_names = X_encoded.columns

# Gain-based feature importance
importance_gain = best_xgb_4.get_booster().get_score(importance_type='gain')

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
print("Top features based on gain Model 4:")
print(importance_gain_df.head(10))

# Plot top features by gain
plt.figure(figsize=(10, 6))
plt.barh(importance_gain_df['Feature'].head(10), importance_gain_df['Gain'].head(10))
plt.xlabel('Gain')
plt.ylabel('Feature')
plt.title('Top 10 Features by Gain Model 4')
plt.gca().invert_yaxis()
plt.show()

# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'Top 10 Features by Gain_model4.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory


# In[ ]:


import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create a SHAP explainer for your model
explainer = shap.Explainer(best_xgb_4)

# Calculate SHAP values for the training set
shap_values = explainer(X_train)

# SHAP summary plot (bar)
shap.summary_plot(shap_values, features=X_train, feature_names=X_encoded.columns, plot_type="bar")
# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'shap_values_bar_model4.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory


# SHAP detailed summary plot
shap.summary_plot(shap_values, features=X_train, feature_names=X_encoded.columns)
# SHAP summary plot (bar)
shap.summary_plot(shap_values, features=X_train, feature_names=X_encoded.columns, plot_type="bar")
# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'shap_values_model4.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory


# In[ ]:


# Create a SHAP explainer for your model
explainer = shap.Explainer(best_xgb_4)

# Calculate SHAP values for the test set
shap_values = explainer(X_test)

# SHAP summary plot (bar)
shap.summary_plot(shap_values, features=X_test, feature_names=X_encoded.columns, plot_type="bar")
output_path = os.path.join(output_dir, 'shap_values_bar_model4_test.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory

# SHAP detailed summary plot
shap.summary_plot(shap_values, features=X_test, feature_names=X_encoded.columns)
output_path = os.path.join(output_dir, 'shap_values_model4_test.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory


# ### Lime

# In[ ]:


import os
import numpy as np
import pandas as pd
import lime
import lime.lime_tabular
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from IPython.display import display, Image

# Predict on the test set
y_pred = best_xgb_4.predict(X_test)
y_pred_proba = best_xgb_4.predict_proba(X_test)

# Separate positive and negative examples with confidence > 80%
positive_indices = np.where((y_pred == 1) & (y_pred_proba[:, 1] > 0.8))[0]
negative_indices = np.where((y_pred == 0) & (y_pred_proba[:, 0] > 0.8))[0]

# Ensure there are examples to choose from
if len(positive_indices) == 0:
    print("No positive examples with confidence > 80% found.")
if len(negative_indices) == 0:
    print("No negative examples with confidence > 80% found.")

# Select multiple instances to explain
# Example: Select the first two positive and negative examples
num_examples = 2
positive_instance_indices = positive_indices[:num_examples] if len(positive_indices) >= num_examples else positive_indices
negative_instance_indices = negative_indices[:num_examples] if len(negative_indices) >= num_examples else negative_indices

# Initialize LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train.values,
    feature_names=X_encoded.columns,
    class_names=np.unique(Y).astype(str),
    discretize_continuous=True
)

# Function to display and save explanations
def display_and_save_explanations(instance_indices, label):
    for idx in instance_indices:
        instance = X_test.iloc[idx].values  # Access the instance using iloc for positional indexing
        confidence = y_pred_proba[idx][1] if y_pred[idx] == 1 else y_pred_proba[idx][0]
        print(f"Confidence for the {label} instance {idx}: {confidence:.2f}")
        exp = explainer.explain_instance(instance, best_xgb_4.predict_proba, num_features=10)
        
        # Display in the notebook
        exp.show_in_notebook(show_table=True, show_all=False)
        
        # Save the explanation as an image
        fig = exp.as_pyplot_figure()
        plt.tight_layout()
        image_path = os.path.join(output_dir, f'lime_explanation_{label}_best_xgb_4_{idx}.png')
        fig.savefig(image_path, format='png', bbox_inches='tight')
        plt.close(fig)
        print(f'Saved explanation to {image_path}')
        
        # Display the saved image in the notebook
        display(Image(filename=image_path))

# Display and save explanations for positive instances
if len(positive_instance_indices) > 0:
    print("Positive Instances with Confidence > 80%:")
    display_and_save_explanations(positive_instance_indices, 'positive')

# Display and save explanations for negative instances
if len(negative_instance_indices) > 0:
    print("Negative Instances with Confidence > 80%:")
    display_and_save_explanations(negative_instance_indices, 'negative')


# ### Experiment 5: Similar to previous one but with lasso and ridge
# 

# In[19]:


import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, make_scorer, confusion_matrix, accuracy_score
import pickle

# Custom scoring function for specificity
def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

# Calculate scale_pos_weight
scale_pos_weight = float(sum(y_train == 0)) / sum(y_train == 1)

# Define a parameter grid with more granular alpha and lambda values
param_dist = {
    'n_estimators': [100, 200, 300, 500, 1000],
    'max_depth': [4, 6, 8, 10],
    'min_child_weight': [10, 20, 30, 40, 50],
    'scale_pos_weight': [scale_pos_weight],
    'subsample': [0.5, 0.8],
    'colsample_bytree': [0.1, 0.2, 0.3, 0.5],
    'learning_rate': [0.0001, 0.001, 0.01, 0.1],
    'gamma': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 1],
    'alpha': [0, 0.01, 0.1, 0.5, 1, 2, 5, 10],  # L1 regularization
    'lambda': [0, 0.01, 0.1, 0.5, 1, 2, 5, 10],  # L2 regularization
}

# XGBoost classifier with GPU support
xgb_model = xgb.XGBClassifier(tree_method='hist', device='cuda', use_label_encoder=False, random_state=42)

# Stratified K-Fold cross-validation
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
outer_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Define custom scoring functions
scoring = {
    'roc_auc': 'roc_auc',
    'recall': 'recall',
    'precision': 'precision',
    'f1': 'f1',
    'specificity': make_scorer(specificity_score),
    'accuracy': 'accuracy'
}

# Randomized search
random_search_5 = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_dist, cv=inner_cv, n_iter=10, scoring='roc_auc', n_jobs=2, verbose=2, random_state=42)

# Nested cross-validation
nested_cv_results = cross_validate(random_search_5, X_train, y_train, cv=outer_cv, scoring=scoring, n_jobs=2, verbose=1, return_train_score=True)

# Print the mean and standard deviation of the nested cross-validation scores
print(f'Nested CV ROC AUC Model 5: {nested_cv_results["test_roc_auc"].mean():.4f} ± {nested_cv_results["test_roc_auc"].std():.4f}')
print(f'Nested CV Recall Model 5: {nested_cv_results["test_recall"].mean():.4f} ± {nested_cv_results["test_recall"].std():.4f}')
print(f'Nested CV Precision Model 5: {nested_cv_results["test_precision"].mean():.4f} ± {nested_cv_results["test_precision"].std():.4f}')
print(f'Nested CV F1 Model 5: {nested_cv_results["test_f1"].mean():.4f} ± {nested_cv_results["test_f1"].std():.4f}')
print(f'Nested CV Specificity Model 5: {nested_cv_results["test_specificity"].mean():.4f} ± {nested_cv_results["test_specificity"].std():.4f}')
print(f'Nested CV Accuracy Model 5: {nested_cv_results["test_accuracy"].mean():.4f} ± {nested_cv_results["test_accuracy"].std():.4f}')

# Fit the model
random_search_5.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_test, y_test)], verbose=10)

# Save the final model
def save_checkpoint(model, filename="checkpoint.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(model, f)
save_checkpoint(random_search_5, "final_model.pkl")

# Best hyperparameters
best_params_5 = random_search_5.best_params_
print(f'Best hyperparameters Model 5: {best_params_5}')

# Test set evaluation
best_xgb_5 = random_search_5.best_estimator_

y_pred_proba = best_xgb_5.predict_proba(X_test)[:, 1]
test_roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f'Test set ROC AUC Model 5: {test_roc_auc:.4f}')

# Additional test set metrics
y_pred = best_xgb_5.predict(X_test)
test_recall = recall_score(y_test, y_pred)
test_precision = precision_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred)
test_specificity = specificity_score(y_test, y_pred)

print(f'Test set Recall Model 5: {test_recall:.4f}')
print(f'Test set Precision Model 5: {test_precision:.4f}')
print(f'Test set F1 Model 5: {test_f1:.4f}')
print(f'Test set Specificity Model 5: {test_specificity:.4f}')


# In[ ]:


print(f'Best hyperparameters Model 5: {best_params_5}')


# In[ ]:


print(f'Test set Recall Model 5: {test_recall:.4f}')
print(f'Test set Precision Model 5: {test_precision:.4f}')
print(f'Test set F1 Model 5: {test_f1:.4f}')
print(f'Test set Specificity Model 5: {test_specificity:.4f}')


# In[ ]:


from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay, recall_score, precision_score, f1_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# make predictions using the best model from cross validation in test dataset
y_pred= best_xgb_5.predict(X_test)

# Classification report
print("Classification Report Model 5:")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("Confusion Matrix Model 5:")
ConfusionMatrixDisplay.from_estimator(best_xgb_5, X_test, y_test, cmap=plt.cm.Blues)
plt.show()

# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'Confusion Matrix_model5.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Model 5: {accuracy:.2f}")

# Recall score (sensitivity)
recall = recall_score(y_test, y_pred)
print(f"Recall (Sensitivity) Model 5: {recall:.2f}")

# Specificity
specificity = recall_score(y_test, y_pred, pos_label=0)
print(f"Specificity Model 5: {specificity:.2f}")

# Precision
precision = precision_score(y_test, y_pred)
print(f"Precision: {precision:.2f}")

# F1 score
f1 = f1_score(y_test, y_pred)
print(f"F1 Score Model 5: {f1:.2f}")

# ROC AUC
y_pred_proba = best_xgb_5.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC AUC Model 5: {roc_auc:.2f}")

# Plot ROC Curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Model 5')
plt.legend(loc='lower right')
plt.show()

# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'Receiver Operating Characteristic_model5.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory

#this is a new item
# Precision-Recall Curve
precisions, recalls, _ = precision_recall_curve(y_test, y_pred)
plt.figure()
plt.plot(recalls, precisions, color='blue', lw=2, label=f'PR curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve Model 5')
plt.legend(loc='lower left')
plt.show()

# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'Precision-Recall Curve_model5.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory


# In[ ]:


import matplotlib.pyplot as plt

# Extract feature names from X_encoded
feature_names = X_encoded.columns

# Gain-based feature importance
importance_gain = best_xgb_5.get_booster().get_score(importance_type='gain')

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
print("Top features based on gain Model 5:")
print(importance_gain_df.head(10))

# Plot top features by gain
plt.figure(figsize=(10, 6))
plt.barh(importance_gain_df['Feature'].head(10), importance_gain_df['Gain'].head(10))
plt.xlabel('Gain')
plt.ylabel('Feature')
plt.title('Top 10 Features by Gain Model 5')
plt.gca().invert_yaxis()
plt.show()

# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'Top 10 Features by Gain_model5.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory



# ### Shap

# In[ ]:


import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create a SHAP explainer for your model
explainer = shap.Explainer(best_xgb_5)

# Calculate SHAP values for the training set
shap_values = explainer(X_train)

# Get the mean absolute SHAP values for each feature
shap_importance = pd.DataFrame(list(zip(X_encoded.columns, np.abs(shap_values.values).mean(axis=0))), columns=['Feature', 'SHAP Value'])
shap_importance = shap_importance.sort_values(by='SHAP Value', ascending=False)

# Display the top features
top_features = shap_importance.head(20)['Feature'].tolist()
print("Top features based on SHAP values:", top_features)

# SHAP summary plot (bar)
shap.summary_plot(shap_values, features=X_train, feature_names=X_encoded.columns, plot_type="bar")
# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'shap_values_bar_model5.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory

# SHAP detailed summary plot
shap.summary_plot(shap_values, features=X_train, feature_names=X_encoded.columns)
# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'shap_values_model5.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory


# In[ ]:


# Create a SHAP explainer for your model
explainer = shap.Explainer(best_xgb_5)

# Calculate SHAP values for the test set
shap_values = explainer(X_test)

# SHAP summary plot (bar)
shap.summary_plot(shap_values, features=X_test, feature_names=X_encoded.columns, plot_type="bar")
output_path = os.path.join(output_dir, 'shap_values_bar_model5_test.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory

# SHAP detailed summary plot
shap.summary_plot(shap_values, features=X_test, feature_names=X_encoded.columns)
output_path = os.path.join(output_dir, 'shap_values_model5_test.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory


# ### Lime

# In[ ]:


import os
import numpy as np
import pandas as pd
import lime
import lime.lime_tabular
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from IPython.display import display, Image

# Predict on the test set
y_pred = best_xgb_5.predict(X_test)
y_pred_proba = best_xgb_5.predict_proba(X_test)

# Separate positive and negative examples with confidence > 80%
positive_indices = np.where((y_pred == 1) & (y_pred_proba[:, 1] > 0.8))[0]
negative_indices = np.where((y_pred == 0) & (y_pred_proba[:, 0] > 0.8))[0]

# Ensure there are examples to choose from
if len(positive_indices) == 0:
    print("No positive examples with confidence > 80% found.")
if len(negative_indices) == 0:
    print("No negative examples with confidence > 80% found.")

# Select multiple instances to explain
# Example: Select the first two positive and negative examples
num_examples = 2
positive_instance_indices = positive_indices[:num_examples] if len(positive_indices) >= num_examples else positive_indices
negative_instance_indices = negative_indices[:num_examples] if len(negative_indices) >= num_examples else negative_indices

# Initialize LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train.values,
    feature_names=X_encoded.columns,
    class_names=np.unique(Y).astype(str),
    discretize_continuous=True
)

# Function to display and save explanations
def display_and_save_explanations(instance_indices, label):
    for idx in instance_indices:
        instance = X_test.iloc[idx].values  # Access the instance using iloc for positional indexing
        confidence = y_pred_proba[idx][1] if y_pred[idx] == 1 else y_pred_proba[idx][0]
        print(f"Confidence for the {label} instance {idx}: {confidence:.2f}")
        exp = explainer.explain_instance(instance, best_xgb_5.predict_proba, num_features=10)
        
        # Display in the notebook
        exp.show_in_notebook(show_table=True, show_all=False)
        
        # Save the explanation as an image
        fig = exp.as_pyplot_figure()
        plt.tight_layout()
        image_path = os.path.join(output_dir, f'lime_explanation_{label}_best_xgb_5_{idx}.png')
        fig.savefig(image_path, format='png', bbox_inches='tight')
        plt.close(fig)
        print(f'Saved explanation to {image_path}')
        
        # Display the saved image in the notebook
        display(Image(filename=image_path))

# Display and save explanations for positive instances
if len(positive_instance_indices) > 0:
    print("Positive Instances with Confidence > 80%:")
    display_and_save_explanations(positive_instance_indices, 'positive')

# Display and save explanations for negative instances
if len(negative_instance_indices) > 0:
    print("Negative Instances with Confidence > 80%:")
    display_and_save_explanations(negative_instance_indices, 'negative')


# ### Experiment 6: Oversampling + best parameter model 5

# In[ ]:


import numpy as np
import xgboost as xgb
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, make_scorer, confusion_matrix
import pickle

# Custom scoring function for specificity
def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

# Apply RandomOverSampler to oversample the minority class
ros = RandomOverSampler(random_state=42)
X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)

# Initialize the XGBoost classifier with the best parameters
best_xgb_6 = xgb.XGBClassifier(**best_params_5)

# Fit the model to the oversampled training data
best_xgb_6.fit(X_train_ros, y_train_ros, early_stopping_rounds=10, eval_set=[(X_test, y_test)], verbose=10)

# Save the final model
with open("final_model.pkl", "wb") as f:
    pickle.dump(best_xgb, f)

# Test set evaluation
y_pred_proba = best_xgb_6.predict_proba(X_test)[:, 1]
test_roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f'Test set ROC AUC Model 6: {test_roc_auc:.4f}')

# Additional test set metrics
y_pred = best_xgb_6.predict(X_test)
test_recall = recall_score(y_test, y_pred)
test_precision = precision_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred)
test_specificity = specificity_score(y_test, y_pred)

print(f'Test set Recall Model 6: {test_recall:.4f}')
print(f'Test set Precision Model 6: {test_precision:.4f}')
print(f'Test set F1 Model 6: {test_f1:.4f}')
print(f'Test set Specificity Model 6: {test_specificity:.4f}')


# In[ ]:


from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay, recall_score, precision_score, f1_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# make predictions using the best model from cross validation in test dataset
y_pred= best_xgb_6.predict(X_test)

# Classification report
print("Classification Report Model 6:")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("Confusion Matrix Model 6:")
ConfusionMatrixDisplay.from_estimator(best_xgb_6, X_test, y_test, cmap=plt.cm.Blues)
plt.show()

# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'Confusion Matrix_model6.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory


# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Model 6: {accuracy:.2f}")

# Recall score (sensitivity)
recall = recall_score(y_test, y_pred)
print(f"Recall (Sensitivity) Model 6: {recall:.2f}")

# Specificity
specificity = recall_score(y_test, y_pred, pos_label=0)
print(f"Specificity Model 6: {specificity:.2f}")

# Precision
precision = precision_score(y_test, y_pred)
print(f"Precision Model 6: {precision:.2f}")

# F1 score
f1 = f1_score(y_test, y_pred)
print(f"F1 Score Model 6: {f1:.2f}")

# ROC AUC
y_pred_proba = best_xgb_6.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC AUC Model 6: {roc_auc:.2f}")

# Plot ROC Curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Model 6')
plt.legend(loc='lower right')
plt.show()

# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'Receiver Operating Characteristic_model6.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory

#this is a new item
# Precision-Recall Curve
precisions, recalls, _ = precision_recall_curve(y_test, y_pred)
plt.figure()
plt.plot(recalls, precisions, color='blue', lw=2, label=f'PR curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve Model 6')
plt.legend(loc='lower left')
plt.show()

# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'Precision-Recall Curve_model6.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory


# In[ ]:


import matplotlib.pyplot as plt

# Extract feature names from X_encoded
feature_names = X_encoded.columns

# Gain-based feature importance
importance_gain = best_xgb_6.get_booster().get_score(importance_type='gain')

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
print("Top features based on gain Model 6:")
print(importance_gain_df.head(10))

# Plot top features by gain
plt.figure(figsize=(10, 6))
plt.barh(importance_gain_df['Feature'].head(10), importance_gain_df['Gain'].head(10))
plt.xlabel('Gain')
plt.ylabel('Feature')
plt.title('Top 10 Features by Gain Model 6')
plt.gca().invert_yaxis()
plt.show()

# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'Top 10 Features by Gain_model6.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory


# In[ ]:


import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create a SHAP explainer for your model
explainer = shap.Explainer(best_xgb_6)

# Calculate SHAP values for the training set
shap_values = explainer(X_train)

# SHAP summary plot (bar)
shap.summary_plot(shap_values, features=X_train, feature_names=X_encoded.columns, plot_type="bar")
# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'shap_bar_model6.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory


# SHAP detailed summary plot
shap.summary_plot(shap_values, features=X_train, feature_names=X_encoded.columns)
# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'shap_model6.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory


# In[ ]:


# Create a SHAP explainer for your model
explainer = shap.Explainer(best_xgb_6)

# Calculate SHAP values for the test set
shap_values = explainer(X_test)

# SHAP summary plot (bar)
shap.summary_plot(shap_values, features=X_test, feature_names=X_encoded.columns, plot_type="bar")
output_path = os.path.join(output_dir, 'shap_values_bar_model6_test.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory

# SHAP detailed summary plot
shap.summary_plot(shap_values, features=X_test, feature_names=X_encoded.columns)
output_path = os.path.join(output_dir, 'shap_values_model6_test.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory


# ### Lime 

# In[ ]:


import os
import numpy as np
import pandas as pd
import lime
import lime.lime_tabular
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from IPython.display import display, Image

# Predict on the test set
y_pred = best_xgb_6.predict(X_test)
y_pred_proba = best_xgb_6.predict_proba(X_test)

# Separate positive and negative examples with confidence > 80%
positive_indices = np.where((y_pred == 1) & (y_pred_proba[:, 1] > 0.8))[0]
negative_indices = np.where((y_pred == 0) & (y_pred_proba[:, 0] > 0.8))[0]

# Ensure there are examples to choose from
if len(positive_indices) == 0:
    print("No positive examples with confidence > 80% found.")
if len(negative_indices) == 0:
    print("No negative examples with confidence > 80% found.")

# Select multiple instances to explain
# Example: Select the first two positive and negative examples
num_examples = 2
positive_instance_indices = positive_indices[:num_examples] if len(positive_indices) >= num_examples else positive_indices
negative_instance_indices = negative_indices[:num_examples] if len(negative_indices) >= num_examples else negative_indices

# Initialize LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train.values,
    feature_names=X_encoded.columns,
    class_names=np.unique(Y).astype(str),
    discretize_continuous=True
)

# Function to display and save explanations
def display_and_save_explanations(instance_indices, label):
    for idx in instance_indices:
        instance = X_test.iloc[idx].values  # Access the instance using iloc for positional indexing
        confidence = y_pred_proba[idx][1] if y_pred[idx] == 1 else y_pred_proba[idx][0]
        print(f"Confidence for the {label} instance {idx}: {confidence:.2f}")
        exp = explainer.explain_instance(instance, best_xgb_6.predict_proba, num_features=10)
        
        # Display in the notebook
        exp.show_in_notebook(show_table=True, show_all=False)
        
        # Save the explanation as an image
        fig = exp.as_pyplot_figure()
        plt.tight_layout()
        image_path = os.path.join(output_dir, f'lime_explanation_{label}_best_xgb_6_{idx}.png')
        fig.savefig(image_path, format='png', bbox_inches='tight')
        plt.close(fig)
        print(f'Saved explanation to {image_path}')
        
        # Display the saved image in the notebook
        display(Image(filename=image_path))

# Display and save explanations for positive instances
if len(positive_instance_indices) > 0:
    print("Positive Instances with Confidence > 80%:")
    display_and_save_explanations(positive_instance_indices, 'positive')

# Display and save explanations for negative instances
if len(negative_instance_indices) > 0:
    print("Negative Instances with Confidence > 80%:")
    display_and_save_explanations(negative_instance_indices, 'negative')


# ### Experiment 7: Undersampling + best parameter model 5 

# In[ ]:


import numpy as np
import xgboost as xgb
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, make_scorer, confusion_matrix
import pickle

# Custom scoring function for specificity
def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

# Apply RandomUnderSampler to undersample the majority class
rus = RandomUnderSampler(random_state=42)
X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)

# Initialize the XGBoost classifier with the best parameters
best_xgb_7 = xgb.XGBClassifier(**best_params_5)

# Fit the model to the undersampled training data
best_xgb_7.fit(X_train_rus, y_train_rus, early_stopping_rounds=10, eval_set=[(X_test, y_test)], verbose=10)

# Save the final model
with open("final_model.pkl", "wb") as f:
    pickle.dump(best_xgb, f)

# Test set evaluation
y_pred_proba = best_xgb_7.predict_proba(X_test)[:, 1]
test_roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f'Test set ROC AUC Model 7: {test_roc_auc:.4f}')

# Additional test set metrics
y_pred = best_xgb_7.predict(X_test)
test_recall = recall_score(y_test, y_pred)
test_precision = precision_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred)
test_specificity = specificity_score(y_test, y_pred)

print(f'Test set Recall Model 7: {test_recall:.4f}')
print(f'Test set Precision Model 7: {test_precision:.4f}')
print(f'Test set F1 Model 7: {test_f1:.4f}')
print(f'Test set Specificity Model 7: {test_specificity:.4f}')


# ### Experiment 8: Hyperparameter search with oversample 

# In[ ]:


import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, make_scorer, confusion_matrix
import pickle

# Custom scoring function for specificity
def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

# Apply RandomOverSampler to oversample the minority class
ros = RandomOverSampler(random_state=42)
X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)

# Define a parameter grid
param_dist = {
    'n_estimators': [100, 200, 300, 500, 1000],
    'max_depth': [4, 6, 8, 10],
    'min_child_weight': [10, 20, 30, 40, 50],
    'subsample': [0.5, 0.8],
    'colsample_bytree': [0.1, 0.2, 0.3, 0.5],
    'learning_rate': [0.0001, 0.001, 0.01, 0.1],
    'gamma': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 1],
}

# XGBoost classifier with GPU support
xgb_model = xgb.XGBClassifier(tree_method='hist', device='cuda', use_label_encoder=False, random_state=42)

# Stratified K-Fold cross-validation
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
outer_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Define custom scoring functions
scoring = {
    'roc_auc': 'roc_auc',
    'recall': 'recall',
    'precision': 'precision',
    'f1': 'f1',
    'specificity': make_scorer(specificity_score)
}

# Randomized search
random_search_8 = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_dist, cv=inner_cv, n_iter=20, scoring='roc_auc', n_jobs=2, verbose=2, random_state=42)

# Function to save progress
def save_checkpoint(model, filename="checkpoint.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(model, f)

# Nested cross-validation
print("Starting nested cross-validation...")
nested_cv_results = cross_validate(random_search_8, X_train_ros, y_train_ros, cv=outer_cv, scoring=scoring, n_jobs=2, verbose=1, return_train_score=True)

# Save checkpoint after cross-validation
save_checkpoint(random_search_8, "random_search_checkpoint.pkl")

# Print the mean and standard deviation of the nested cross-validation scores
print(f'Nested CV ROC AUC Model 8: {nested_cv_results["test_roc_auc"].mean():.4f} ± {nested_cv_results["test_roc_auc"].std():.4f}')
print(f'Nested CV Recall Model 8: {nested_cv_results["test_recall"].mean():.4f} ± {nested_cv_results["test_recall"].std():.4f}')
print(f'Nested CV Precision Model 8: {nested_cv_results["test_precision"].mean():.4f} ± {nested_cv_results["test_precision"].std():.4f}')
print(f'Nested CV F1 Model 8: {nested_cv_results["test_f1"].mean():.4f} ± {nested_cv_results["test_f1"].std():.4f}')
print(f'Nested CV Specificity Model 8: {nested_cv_results["test_specificity"].mean():.4f} ± {nested_cv_results["test_specificity"].std():.4f}')

# Fit the model with the entire training set
print("Fitting the best model...")
random_search_8.fit(X_train_ros, y_train_ros, early_stopping_rounds=10, eval_set=[(X_test, y_test)], verbose=10)

# Save the final model
save_checkpoint(random_search_8, "final_model.pkl")

# Best hyperparameters
best_params_8 = random_search_8.best_params_
print(f'Best hyperparameters Model 8: {best_params_8}')

# Test set evaluation
best_xgb_8 = random_search_8.best_estimator_

y_pred_proba = best_xgb_8.predict_proba(X_test)[:, 1]
test_roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f'Test set ROC AUC Model 8: {test_roc_auc:.4f}')

# Additional test set metrics
y_pred = best_xgb_8.predict(X_test)
test_recall = recall_score(y_test, y_pred)
test_precision = precision_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred)
test_specificity = specificity_score(y_test, y_pred)

print(f'Test set Recall Model 8: {test_recall:.4f}')
print(f'Test set Precision Model 8: {test_precision:.4f}')
print(f'Test set F1 Model 8: {test_f1:.4f}')
print(f'Test set Specificity Model 8: {test_specificity:.4f}')


# In[ ]:


from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay, recall_score, precision_score, f1_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# make predictions using the best model from cross validation in test dataset
y_pred= best_xgb_8.predict(X_test)

# Classification report
print("Classification Report Model 8:")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("Confusion Matrix Model 8:")
ConfusionMatrixDisplay.from_estimator(best_xgb_8, X_test, y_test, cmap=plt.cm.Blues)
plt.show()

# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'Confusion Matrix_model8.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Model 8: {accuracy:.2f}")

# Recall score (sensitivity)
recall = recall_score(y_test, y_pred)
print(f"Recall (Sensitivity) Model 8: {recall:.2f}")

# Specificity
specificity = recall_score(y_test, y_pred, pos_label=0)
print(f"Specificity Model 8: {specificity:.2f}")

# Precision
precision = precision_score(y_test, y_pred)
print(f"Precision Model 8: {precision:.2f}")

# F1 score
f1 = f1_score(y_test, y_pred)
print(f"F1 Score Model 8: {f1:.2f}")

# ROC AUC
y_pred_proba = best_xgb_8.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC AUC Model 8: {roc_auc:.2f}")

# Plot ROC Curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Model 8')
plt.legend(loc='lower right')
plt.show()

# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'Receiver Operating Characteristic_model8.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory

#this is a new item
# Precision-Recall Curve
precisions, recalls, _ = precision_recall_curve(y_test, y_pred)
plt.figure()
plt.plot(recalls, precisions, color='blue', lw=2, label=f'PR curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve Model 8')
plt.legend(loc='lower left')
plt.show()

# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'Precision-Recall Curve_model8.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory


# In[ ]:


import matplotlib.pyplot as plt

# Extract feature names from X_encoded
feature_names = X_encoded.columns

# Gain-based feature importance
importance_gain = best_xgb_8.get_booster().get_score(importance_type='gain')

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
print("Top features based on gain Model 8:")
print(importance_gain_df.head(10))

# Plot top features by gain
plt.figure(figsize=(10, 6))
plt.barh(importance_gain_df['Feature'].head(10), importance_gain_df['Gain'].head(10))
plt.xlabel('Gain')
plt.ylabel('Feature')
plt.title('Top 10 Features by Gain Model 8')
plt.gca().invert_yaxis()
plt.show()

# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'Top 10 Features by Gain_model8.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory


# In[ ]:


import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create a SHAP explainer for your model
explainer = shap.Explainer(best_xgb_8)

# Calculate SHAP values for the training set
shap_values = explainer(X_train)

# SHAP summary plot (bar)
shap.summary_plot(shap_values, features=X_train, feature_names=X_encoded.columns, plot_type="bar")
# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'shap_values_bar_model8.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory


# SHAP detailed summary plot
shap.summary_plot(shap_values, features=X_train, feature_names=X_encoded.columns)
# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'shap_values_model8.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory


# In[ ]:


# Create a SHAP explainer for your model
explainer = shap.Explainer(best_xgb_8)

# Calculate SHAP values for the test set
shap_values = explainer(X_test)

# SHAP summary plot (bar)
shap.summary_plot(shap_values, features=X_test, feature_names=X_encoded.columns, plot_type="bar")
output_path = os.path.join(output_dir, 'shap_values_bar_model8_test.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory

# SHAP detailed summary plot
shap.summary_plot(shap_values, features=X_test, feature_names=X_encoded.columns)
output_path = os.path.join(output_dir, 'shap_values_model8_test.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory


# In[ ]:


import os
import numpy as np
import pandas as pd
import lime
import lime.lime_tabular
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from IPython.display import display, Image

# Predict on the test set
y_pred = best_xgb_8.predict(X_test)
y_pred_proba = best_xgb_8.predict_proba(X_test)

# Separate positive and negative examples with confidence > 80%
positive_indices = np.where((y_pred == 1) & (y_pred_proba[:, 1] > 0.8))[0]
negative_indices = np.where((y_pred == 0) & (y_pred_proba[:, 0] > 0.8))[0]

# Ensure there are examples to choose from
if len(positive_indices) == 0:
    print("No positive examples with confidence > 80% found.")
if len(negative_indices) == 0:
    print("No negative examples with confidence > 80% found.")

# Select multiple instances to explain
# Example: Select the first two positive and negative examples
num_examples = 2
positive_instance_indices = positive_indices[:num_examples] if len(positive_indices) >= num_examples else positive_indices
negative_instance_indices = negative_indices[:num_examples] if len(negative_indices) >= num_examples else negative_indices

# Initialize LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train.values,
    feature_names=X_encoded.columns,
    class_names=np.unique(Y).astype(str),
    discretize_continuous=True
)

# Function to display and save explanations
def display_and_save_explanations(instance_indices, label):
    for idx in instance_indices:
        instance = X_test.iloc[idx].values  # Access the instance using iloc for positional indexing
        confidence = y_pred_proba[idx][1] if y_pred[idx] == 1 else y_pred_proba[idx][0]
        print(f"Confidence for the {label} instance {idx}: {confidence:.2f}")
        exp = explainer.explain_instance(instance, best_xgb_8.predict_proba, num_features=10)
        
        # Display in the notebook
        exp.show_in_notebook(show_table=True, show_all=False)
        
        # Save the explanation as an image
        fig = exp.as_pyplot_figure()
        plt.tight_layout()
        image_path = os.path.join(output_dir, f'lime_explanation_{label}_best_xgb_8_{idx}.png')
        fig.savefig(image_path, format='png', bbox_inches='tight')
        plt.close(fig)
        print(f'Saved explanation to {image_path}')
        
        # Display the saved image in the notebook
        display(Image(filename=image_path))

# Display and save explanations for positive instances
if len(positive_instance_indices) > 0:
    print("Positive Instances with Confidence > 80%:")
    display_and_save_explanations(positive_instance_indices, 'positive')

# Display and save explanations for negative instances
if len(negative_instance_indices) > 0:
    print("Negative Instances with Confidence > 80%:")
    display_and_save_explanations(negative_instance_indices, 'negative')


# ### experiment 9: Hyperparameter search with oversample + L1 and L2 

# In[ ]:


import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, make_scorer, confusion_matrix
import pickle

# Custom scoring function for specificity
def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

# Apply RandomOverSampler to oversample the minority class
ros = RandomOverSampler(random_state=42)
X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)

# Define a parameter grid
param_dist = {
    'n_estimators': [100, 200, 300, 500, 1000],
    'max_depth': [4, 6, 8, 10],
    'min_child_weight': [10, 20, 30, 40, 50],
    'subsample': [0.5, 0.8],
    'colsample_bytree': [0.1, 0.2, 0.3, 0.5],
    'learning_rate': [0.0001, 0.001, 0.01, 0.1],
    'gamma': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 1],
    'alpha': [0, 0.01, 0.1, 0.5, 1, 2, 5, 10],  # L1 regularization
    'lambda': [0, 0.01, 0.1, 0.5, 1, 2, 5, 10],  # L2 regularization
}

# XGBoost classifier with GPU support
xgb_model = xgb.XGBClassifier(tree_method='hist', device='cuda', use_label_encoder=False, random_state=42)

# Stratified K-Fold cross-validation
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
outer_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Define custom scoring functions
scoring = {
    'roc_auc': 'roc_auc',
    'recall': 'recall',
    'precision': 'precision',
    'f1': 'f1',
    'specificity': make_scorer(specificity_score)
}

# Randomized search
random_search_9 = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_dist, cv=inner_cv, n_iter=20, scoring='roc_auc', n_jobs=2, verbose=2, random_state=42)

# Function to save progress
def save_checkpoint(model, filename="checkpoint.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(model, f)

# Nested cross-validation
print("Starting nested cross-validation...")
nested_cv_results = cross_validate(random_search_9, X_train_ros, y_train_ros, cv=outer_cv, scoring=scoring, n_jobs=2, verbose=1, return_train_score=True)

# Save checkpoint after cross-validation
save_checkpoint(random_search_9, "random_search_checkpoint.pkl")

# Print the mean and standard deviation of the nested cross-validation scores
print(f'Nested CV ROC AUC Model 9: {nested_cv_results["test_roc_auc"].mean():.4f} ± {nested_cv_results["test_roc_auc"].std():.4f}')
print(f'Nested CV Recall Model 9: {nested_cv_results["test_recall"].mean():.4f} ± {nested_cv_results["test_recall"].std():.4f}')
print(f'Nested CV Precision Model 9: {nested_cv_results["test_precision"].mean():.4f} ± {nested_cv_results["test_precision"].std():.4f}')
print(f'Nested CV F1 Model 9: {nested_cv_results["test_f1"].mean():.4f} ± {nested_cv_results["test_f1"].std():.4f}')
print(f'Nested CV Specificity Model 9: {nested_cv_results["test_specificity"].mean():.4f} ± {nested_cv_results["test_specificity"].std():.4f}')

# Fit the model with the entire training set
print("Fitting the best model...")
random_search_9.fit(X_train_ros, y_train_ros, early_stopping_rounds=10, eval_set=[(X_test, y_test)], verbose=10)

# Save the final model
save_checkpoint(random_search_9, "final_model.pkl")

# Best hyperparameters
best_params_9 = random_search_9.best_params_
print(f'Best hyperparameters Model 9: {best_params_9}')

# Test set evaluation
best_xgb_9 = random_search_9.best_estimator_

y_pred_proba = best_xgb_9.predict_proba(X_test)[:, 1]
test_roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f'Test set ROC AUC Model 9: {test_roc_auc:.4f}')

# Additional test set metrics
y_pred = best_xgb_9.predict(X_test)
test_recall = recall_score(y_test, y_pred)
test_precision = precision_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred)
test_specificity = specificity_score(y_test, y_pred)

print(f'Test set Recall Model 9: {test_recall:.4f}')
print(f'Test set Precision Model 9: {test_precision:.4f}')
print(f'Test set F1 Model 9: {test_f1:.4f}')
print(f'Test set Specificity Model 9: {test_specificity:.4f}')


# In[ ]:


from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay, recall_score, precision_score, f1_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# make predictions using the best model from cross validation in test dataset
y_pred= best_xgb_9.predict(X_test)

# Classification report
print("Classification Report Model 9:")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("Confusion Matrix Model 9:")
ConfusionMatrixDisplay.from_estimator(best_xgb_9, X_test, y_test, cmap=plt.cm.Blues)
plt.show()

# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'Confusion Matrix_model9.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Model 9: {accuracy:.2f}")

# Recall score (sensitivity)
recall = recall_score(y_test, y_pred)
print(f"Recall (Sensitivity) Model 9: {recall:.2f}")

# Specificity
specificity = recall_score(y_test, y_pred, pos_label=0)
print(f"Specificity Model 9: {specificity:.2f}")

# Precision
precision = precision_score(y_test, y_pred)
print(f"Precision Model 9: {precision:.2f}")

# F1 score
f1 = f1_score(y_test, y_pred)
print(f"F1 Score Model 9: {f1:.2f}")

# ROC AUC
y_pred_proba = best_xgb_9.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC AUC Model 9: {roc_auc:.2f}")

# Plot ROC Curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Model 9')
plt.legend(loc='lower right')
plt.show()

# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'Receiver Operating Characteristic_model9.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory


#this is a new item
# Precision-Recall Curve
precisions, recalls, _ = precision_recall_curve(y_test, y_pred)
plt.figure()
plt.plot(recalls, precisions, color='blue', lw=2, label=f'PR curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve Model 9')
plt.legend(loc='lower left')
plt.show()

# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'Precision-Recall Curve_model9.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory


# In[ ]:


import matplotlib.pyplot as plt

# Extract feature names from X_encoded
feature_names = X_encoded.columns

# Gain-based feature importance
importance_gain = best_xgb_9.get_booster().get_score(importance_type='gain')

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
print("Top features based on gain Model 9:")
print(importance_gain_df.head(10))

# Plot top features by gain
plt.figure(figsize=(10, 6))
plt.barh(importance_gain_df['Feature'].head(10), importance_gain_df['Gain'].head(10))
plt.xlabel('Gain')
plt.ylabel('Feature')
plt.title('Top 10 Features by Gain Model 9')
plt.gca().invert_yaxis()
plt.show()

# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'Top 10 Features by Gain_model9.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory



# In[ ]:


import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create a SHAP explainer for your model
explainer = shap.Explainer(best_xgb_9)

# Calculate SHAP values for the training set
shap_values = explainer(X_train)

# SHAP summary plot (bar)
shap.summary_plot(shap_values, features=X_train, feature_names=X_encoded.columns, plot_type="bar")

# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'shap_values_bar_model9.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory


# SHAP detailed summary plot
shap.summary_plot(shap_values, features=X_train, feature_names=X_encoded.columns)

# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'shap_values_model9.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory


# In[ ]:


# Create a SHAP explainer for your model
explainer = shap.Explainer(best_xgb_9)

# Calculate SHAP values for the test set
shap_values = explainer(X_test)

# SHAP summary plot (bar)
shap.summary_plot(shap_values, features=X_test, feature_names=X_encoded.columns, plot_type="bar")
output_path = os.path.join(output_dir, 'shap_values_bar_model9_test.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory

# SHAP detailed summary plot
shap.summary_plot(shap_values, features=X_test, feature_names=X_encoded.columns)
output_path = os.path.join(output_dir, 'shap_values_model9_test.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory


# In[ ]:


import os
import numpy as np
import pandas as pd
import lime
import lime.lime_tabular
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from IPython.display import display, Image

# Predict on the test set
y_pred = best_xgb_9.predict(X_test)
y_pred_proba = best_xgb_9.predict_proba(X_test)

# Separate positive and negative examples with confidence > 80%
positive_indices = np.where((y_pred == 1) & (y_pred_proba[:, 1] > 0.8))[0]
negative_indices = np.where((y_pred == 0) & (y_pred_proba[:, 0] > 0.8))[0]

# Ensure there are examples to choose from
if len(positive_indices) == 0:
    print("No positive examples with confidence > 80% found.")
if len(negative_indices) == 0:
    print("No negative examples with confidence > 80% found.")

# Select multiple instances to explain
# Example: Select the first two positive and negative examples
num_examples = 2
positive_instance_indices = positive_indices[:num_examples] if len(positive_indices) >= num_examples else positive_indices
negative_instance_indices = negative_indices[:num_examples] if len(negative_indices) >= num_examples else negative_indices

# Initialize LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train.values,
    feature_names=X_encoded.columns,
    class_names=np.unique(Y).astype(str),
    discretize_continuous=True
)

# Function to display and save explanations
def display_and_save_explanations(instance_indices, label):
    for idx in instance_indices:
        instance = X_test.iloc[idx].values  # Access the instance using iloc for positional indexing
        confidence = y_pred_proba[idx][1] if y_pred[idx] == 1 else y_pred_proba[idx][0]
        print(f"Confidence for the {label} instance {idx}: {confidence:.2f}")
        exp = explainer.explain_instance(instance, best_xgb_9.predict_proba, num_features=10)
        
        # Display in the notebook
        exp.show_in_notebook(show_table=True, show_all=False)
        
        # Save the explanation as an image
        fig = exp.as_pyplot_figure()
        plt.tight_layout()
        image_path = os.path.join(output_dir, f'lime_explanation_{label}_best_xgb_9_{idx}.png')
        fig.savefig(image_path, format='png', bbox_inches='tight')
        plt.close(fig)
        print(f'Saved explanation to {image_path}')
        
        # Display the saved image in the notebook
        display(Image(filename=image_path))

# Display and save explanations for positive instances
if len(positive_instance_indices) > 0:
    print("Positive Instances with Confidence > 80%:")
    display_and_save_explanations(positive_instance_indices, 'positive')

# Display and save explanations for negative instances
if len(negative_instance_indices) > 0:
    print("Negative Instances with Confidence > 80%:")
    display_and_save_explanations(negative_instance_indices, 'negative')


# ### Experiment 10: only using feature selection from SHAP (from experiment 5 as best ROC AUC achieved)

# In[ ]:


import imblearn

imblearn.__version__


# In[139]:


import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, make_scorer, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
import pickle

# Features identified by SHAP
selected_features = top_features  # Automatically selected top features

# Ensure the dataset contains only these features
X_selected = X_encoded[selected_features]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, Y, test_size=0.3, random_state=42, stratify=Y)

# Standardize the data after splitting
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Apply RandomOverSampler to oversample the minority class
ros = RandomOverSampler(random_state=42)
X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)

# Define a parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300, 500, 1000],
    'max_depth': [4, 6, 8, 10],
    'min_child_weight': [10, 20, 30, 40, 50],
    'subsample': [0.5, 0.8],
    'colsample_bytree': [0.1, 0.2, 0.3, 0.5],
    'learning_rate': [0.0001, 0.001, 0.01, 0.1],
    'gamma': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 1],
    'alpha': [0, 0.01, 0.1, 0.5, 1, 2, 5, 10],  # L1 regularization
    'lambda': [0, 0.01, 0.1, 0.5, 1, 2, 5, 10],  # L2 regularization
}

# XGBoost classifier with GPU support
xgb_model = xgb.XGBClassifier(tree_method='hist', device='cuda', use_label_encoder=False, random_state=42)

# Stratified K-Fold cross-validation
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
outer_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Define custom scoring functions
scoring = {
    'roc_auc': 'roc_auc',
    'recall': 'recall',
    'precision': 'precision',
    'f1': 'f1',
    'specificity': make_scorer(specificity_score)
}

# Grid search
grid_search_10 = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_grid, cv=inner_cv, n_iter=20, scoring='roc_auc', n_jobs=2, verbose=2, random_state=42)

# Function to save progress
def save_checkpoint(model, filename="checkpoint.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(model, f)

# Nested cross-validation
print("Starting nested cross-validation...")
nested_cv_results = cross_validate(grid_search_10, X_train_ros, y_train_ros, cv=outer_cv, scoring=scoring, n_jobs=2, verbose=1, return_train_score=True)

# Save checkpoint after cross-validation
save_checkpoint(grid_search_10, "grid_search_checkpoint.pkl")

# Print the mean and standard deviation of the nested cross-validation scores
print(f'Nested CV ROC AUC Model 10: {nested_cv_results["test_roc_auc"].mean():.4f} ± {nested_cv_results["test_roc_auc"].std():.4f}')
print(f'Nested CV Recall Model 10: {nested_cv_results["test_recall"].mean():.4f} ± {nested_cv_results["test_recall"].std():.4f}')
print(f'Nested CV Precision Model 10: {nested_cv_results["test_precision"].mean():.4f} ± {nested_cv_results["test_precision"].std():.4f}')
print(f'Nested CV F1 Model 10: {nested_cv_results["test_f1"].mean():.4f} ± {nested_cv_results["test_f1"].std():.4f}')
print(f'Nested CV Specificity Model 10: {nested_cv_results["test_specificity"].mean():.4f} ± {nested_cv_results["test_specificity"].std():.4f}')

# Fit the model with the entire training set
print("Fitting the best model...")
grid_search_10.fit(X_train_ros, y_train_ros, early_stopping_rounds=10, eval_set=[(X_test, y_test)], verbose=10)

# Save the final model
save_checkpoint(grid_search_10, "final_model.pkl")

# Best hyperparameters
best_params_10 = grid_search_10.best_params_
print(f'Best hyperparameters Model 10: {best_params_10}')

# Test set evaluation
best_xgb_10 = grid_search_10.best_estimator_

y_pred_proba = best_xgb_10.predict_proba(X_test)[:, 1]
test_roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f'Test set ROC AUC Model 10: {test_roc_auc:.4f}')

# Additional test set metrics
y_pred = best_xgb_10.predict(X_test)
test_recall = recall_score(y_test, y_pred)
test_precision = precision_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred)
test_specificity = specificity_score(y_test, y_pred)

print(f'Test set Recall Model 10: {test_recall:.4f}')
print(f'Test set Precision Model 10: {test_precision:.4f}')
print(f'Test set F1 Model 10: {test_f1:.4f}')
print(f'Test set Specificity Model 10: {test_specificity:.4f}')


# In[144]:


from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay, recall_score, precision_score, f1_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# make predictions using the best model from cross validation in test dataset
y_pred= best_xgb_10.predict(X_test)

# Classification report
print("Classification Report Model 10:")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("Confusion Matrix Model 10:")
ConfusionMatrixDisplay.from_estimator(best_xgb_10, X_test, y_test, cmap=plt.cm.Blues)
plt.show()

# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'Confusion Matrix_model10.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Model 10: {accuracy:.2f}")

# Recall score (sensitivity)
recall = recall_score(y_test, y_pred)
print(f"Recall (Sensitivity) Model 10: {recall:.2f}")

# Specificity
specificity = recall_score(y_test, y_pred, pos_label=0)
print(f"Specificity Model 10: {specificity:.2f}")

# Precision
precision = precision_score(y_test, y_pred)
print(f"Precision Model 10: {precision:.2f}")

# F1 score
f1 = f1_score(y_test, y_pred)
print(f"F1 Score Model 10: {f1:.2f}")

# ROC AUC
y_pred_proba = best_xgb_10.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC AUC Model 10: {roc_auc:.2f}")

# Plot ROC Curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Model 10')
plt.legend(loc='lower right')
plt.show()

# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'Receiver Operating Characteristic_model10.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memor


#this is a new item
# Precision-Recall Curve
precisions, recalls, _ = precision_recall_curve(y_test, y_pred)
plt.figure()
plt.plot(recalls, precisions, color='blue', lw=2, label=f'PR curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve Model 10')
plt.legend(loc='lower left')
plt.show()


# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'Precision-Recall Curve_model10.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memor


# In[141]:


import matplotlib.pyplot as plt

# Extract feature names from X_encoded
feature_names = X_selected.columns

# Gain-based feature importance
importance_gain = best_xgb_10.get_booster().get_score(importance_type='gain')

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
print("Top features based on gain Model 10:")
print(importance_gain_df.head(10))

# Plot top features by gain
plt.figure(figsize=(10, 6))
plt.barh(importance_gain_df['Feature'].head(10), importance_gain_df['Gain'].head(10))
plt.xlabel('Gain')
plt.ylabel('Feature')
plt.title('Top 10 Features by Gain Model 10')
plt.gca().invert_yaxis()
plt.show()

# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'Top 10 Features by Gain_model10.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory


# In[142]:


import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create a SHAP explainer for your model
explainer = shap.Explainer(best_xgb_10)

# Calculate SHAP values for the training set
shap_values = explainer(X_train)

# SHAP summary plot (bar)
shap.summary_plot(shap_values, features=X_train, feature_names=selected_features, plot_type="bar")
# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'shap_values_bar_model10.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memor


# SHAP detailed summary plot
shap.summary_plot(shap_values, features=X_train, feature_names=selected_features)
# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'shap_values_model10.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory


# In[ ]:


# Create a SHAP explainer for your model
explainer = shap.Explainer(best_xgb_10)

# Calculate SHAP values for the test set
shap_values = explainer(X_test)

# SHAP summary plot (bar)
shap.summary_plot(shap_values, features=X_test, feature_names=selected_features, plot_type="bar")
output_path = os.path.join(output_dir, 'shap_values_bar_model10_test.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory

# SHAP detailed summary plot
shap.summary_plot(shap_values, features=X_test, feature_names=selected_features)
output_path = os.path.join(output_dir, 'shap_values_model10_test.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory


# In[ ]:


import os
import numpy as np
import pandas as pd
import lime
import lime.lime_tabular
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from IPython.display import display, Image

# Predict on the test set
y_pred = best_xgb_10.predict(X_test)
y_pred_proba = best_xgb_10.predict_proba(X_test)

# Separate positive and negative examples with confidence > 80%
positive_indices = np.where((y_pred == 1) & (y_pred_proba[:, 1] > 0.8))[0]
negative_indices = np.where((y_pred == 0) & (y_pred_proba[:, 0] > 0.8))[0]

# Ensure there are examples to choose from
if len(positive_indices) == 0:
    print("No positive examples with confidence > 80% found.")
if len(negative_indices) == 0:
    print("No negative examples with confidence > 80% found.")

# Select multiple instances to explain
# Example: Select the first two positive and negative examples
num_examples = 2
positive_instance_indices = positive_indices[:num_examples] if len(positive_indices) >= num_examples else positive_indices
negative_instance_indices = negative_indices[:num_examples] if len(negative_indices) >= num_examples else negative_indices

# Initialize LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train,
    feature_names=selected_features,
    class_names=np.unique(Y).astype(str),
    discretize_continuous=True
)

# Function to display and save explanations
def display_and_save_explanations(instance_indices, label):
    for idx in instance_indices:
        instance = X_test[idx]  # Use direct indexing since X_test is a numpy array
        confidence = y_pred_proba[idx][1] if y_pred[idx] == 1 else y_pred_proba[idx][0]
        print(f"Confidence for the {label} instance {idx}: {confidence:.2f}")
        exp = explainer.explain_instance(instance, best_xgb_10.predict_proba, num_features=10)
        
        # Display in the notebook
        exp.show_in_notebook(show_table=True, show_all=False)
        
        # Save the explanation as an image
        fig = exp.as_pyplot_figure()
        plt.tight_layout()
        image_path = os.path.join(output_dir, f'lime_explanation_{label}_best_xgb_10_{idx}.png')
        fig.savefig(image_path, format='png', bbox_inches='tight')
        plt.close(fig)
        print(f'Saved explanation to {image_path}')
        
        # Display the saved image in the notebook
        display(Image(filename=image_path))

# Display and save explanations for positive instances
if len(positive_instance_indices) > 0:
    print("Positive Instances with Confidence > 80%:")
    display_and_save_explanations(positive_instance_indices, 'positive')

# Display and save explanations for negative instances
if len(negative_instance_indices) > 0:
    print("Negative Instances with Confidence > 80%:")
    display_and_save_explanations(negative_instance_indices, 'negative')


# ### Experiment 11: like exp 10 but without oversampling the minority class

# In[143]:


import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, make_scorer, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
import pickle

# Features identified by SHAP
selected_features = top_features  # Automatically selected top features

# Ensure the dataset contains only these features
X_selected = X_encoded[selected_features]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, Y, test_size=0.3, random_state=42)

# Standardize the data after splitting
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define a parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300, 500, 1000],
    'max_depth': [4, 6, 8, 10],
    'min_child_weight': [10, 20, 30, 40, 50],
    'subsample': [0.5, 0.8],
    'scale_pos_weight': [scale_pos_weight],
    'colsample_bytree': [0.1, 0.2, 0.3, 0.5],
    'learning_rate': [0.0001, 0.001, 0.01, 0.1],
    'gamma': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 1],
    'alpha': [0, 0.01, 0.1, 0.5, 1, 2, 5, 10],  # L1 regularization
    'lambda': [0, 0.01, 0.1, 0.5, 1, 2, 5, 10],  # L2 regularization
}

# XGBoost classifier with GPU support
xgb_model = xgb.XGBClassifier(tree_method='hist', device='cuda', use_label_encoder=False, random_state=42)

# Stratified K-Fold cross-validation
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
outer_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Define custom scoring functions
scoring = {
    'roc_auc': 'roc_auc',
    'recall': 'recall',
    'precision': 'precision',
    'f1': 'f1',
    'specificity': make_scorer(specificity_score)
}

# Grid search
grid_search_11 = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_grid, cv=inner_cv, n_iter=20, scoring='roc_auc', n_jobs=2, verbose=2, random_state=42)

# Function to save progress
def save_checkpoint(model, filename="checkpoint.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(model, f)

# Nested cross-validation
print("Starting nested cross-validation...")
nested_cv_results = cross_validate(grid_search_11, X_train, y_train, cv=outer_cv, scoring=scoring, n_jobs=2, verbose=1, return_train_score=True)

# Save checkpoint after cross-validation
save_checkpoint(grid_search_11, "grid_search_checkpoint.pkl")

# Print the mean and standard deviation of the nested cross-validation scores
print(f'Nested CV ROC AUC Model 11: {nested_cv_results["test_roc_auc"].mean():.4f} ± {nested_cv_results["test_roc_auc"].std():.4f}')
print(f'Nested CV Recall Model 11: {nested_cv_results["test_recall"].mean():.4f} ± {nested_cv_results["test_recall"].std():.4f}')
print(f'Nested CV Precision Model 11: {nested_cv_results["test_precision"].mean():.4f} ± {nested_cv_results["test_precision"].std():.4f}')
print(f'Nested CV F1 Model 11: {nested_cv_results["test_f1"].mean():.4f} ± {nested_cv_results["test_f1"].std():.4f}')
print(f'Nested CV Specificity Model 11: {nested_cv_results["test_specificity"].mean():.4f} ± {nested_cv_results["test_specificity"].std():.4f}')

# Fit the model with the entire training set
print("Fitting the best model...")
grid_search_11.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_test, y_test)], verbose=10)

# Save the final model
save_checkpoint(grid_search_11, "final_model.pkl")

# Best hyperparameters
best_params_11 = grid_search_11.best_params_
print(f'Best hyperparameters Model 11: {best_params_11}')

# Test set evaluation
best_xgb_11 = grid_search_11.best_estimator_

y_pred_proba = best_xgb_11.predict_proba(X_test)[:, 1]
test_roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f'Test set ROC AUC Model 11: {test_roc_auc:.4f}')

# Additional test set metrics
y_pred = best_xgb_11.predict(X_test)
test_recall = recall_score(y_test, y_pred)
test_precision = precision_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred)
test_specificity = specificity_score(y_test, y_pred)

print(f'Test set Recall Model 11: {test_recall:.4f}')
print(f'Test set Precision Model 11: {test_precision:.4f}')
print(f'Test set F1 Model 11: {test_f1:.4f}')
print(f'Test set Specificity Model 11: {test_specificity:.4f}')


# In[145]:


from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay, recall_score, precision_score, f1_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# make predictions using the best model from cross validation in test dataset
y_pred= best_xgb_11.predict(X_test)

# Classification report
print("Classification Report Model 11:")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("Confusion Matrix Model 11:")
ConfusionMatrixDisplay.from_estimator(best_xgb_11, X_test, y_test, cmap=plt.cm.Blues)
plt.show()

# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'Confusion Matrix_model11.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory


# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Model 11: {accuracy:.2f}")

# Recall score (sensitivity)
recall = recall_score(y_test, y_pred)
print(f"Recall (Sensitivity) Model 11: {recall:.2f}")

# Specificity
specificity = recall_score(y_test, y_pred, pos_label=0)
print(f"Specificity Model 11: {specificity:.2f}")

# Precision
precision = precision_score(y_test, y_pred)
print(f"Precision Model 11: {precision:.2f}")

# F1 score
f1 = f1_score(y_test, y_pred)
print(f"F1 Score Model 11: {f1:.2f}")

# ROC AUC
y_pred_proba = best_xgb_11.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC AUC: {roc_auc:.2f}")

# Plot ROC Curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Model 11')
plt.legend(loc='lower right')
plt.show()

# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'Receiver Operating Characteristic_model11.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory


#this is a new item
# Precision-Recall Curve
precisions, recalls, _ = precision_recall_curve(y_test, y_pred)
plt.figure()
plt.plot(recalls, precisions, color='blue', lw=2, label=f'PR curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve Model 11')
plt.legend(loc='lower left')
plt.show()

# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'Precision-Recall Curve_model11.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory


# In[ ]:


import matplotlib.pyplot as plt

# Extract feature names from X_encoded
feature_names = X_selected.columns

# Gain-based feature importance
importance_gain = best_xgb_11.get_booster().get_score(importance_type='gain')

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
print("Top features based on gain Model 11:")
print(importance_gain_df.head(10))

# Plot top features by gain
plt.figure(figsize=(10, 6))
plt.barh(importance_gain_df['Feature'].head(10), importance_gain_df['Gain'].head(10))
plt.xlabel('Gain')
plt.ylabel('Feature')
plt.title('Top 10 Features by Gain Model 11')
plt.gca().invert_yaxis()
plt.show()

# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'Top 10 Features by Gain_model11.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory


# In[ ]:


import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create a SHAP explainer for your model
explainer = shap.Explainer(best_xgb_11)

# Calculate SHAP values for the training set
shap_values = explainer(X_train)

# SHAP summary plot (bar)
shap.summary_plot(shap_values, features=X_train, feature_names=selected_features, plot_type="bar")
# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'shap_values_bar_model11.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memor

# SHAP detailed summary plot
shap.summary_plot(shap_values, features=X_train, feature_names=selected_features)
# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'shap_values_model11.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory


# In[ ]:


# Create a SHAP explainer for your model
explainer = shap.Explainer(best_xgb_11)

# Calculate SHAP values for the test set
shap_values = explainer(X_test)

# SHAP summary plot (bar)
shap.summary_plot(shap_values, features=X_test, feature_names=selected_features, plot_type="bar")
output_path = os.path.join(output_dir, 'shap_values_bar_model11_test.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory

# SHAP detailed summary plot
shap.summary_plot(shap_values, features=X_test, feature_names=selected_features)
output_path = os.path.join(output_dir, 'shap_values_model11_test.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory


# In[ ]:


import os
import numpy as np
import pandas as pd
import lime
import lime.lime_tabular
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from IPython.display import display, Image

# Predict on the test set
y_pred = best_xgb_11.predict(X_test)
y_pred_proba = best_xgb_11.predict_proba(X_test)

# Separate positive and negative examples with confidence > 80%
positive_indices = np.where((y_pred == 1) & (y_pred_proba[:, 1] > 0.8))[0]
negative_indices = np.where((y_pred == 0) & (y_pred_proba[:, 0] > 0.8))[0]

# Ensure there are examples to choose from
if len(positive_indices) == 0:
    print("No positive examples with confidence > 80% found.")
if len(negative_indices) == 0:
    print("No negative examples with confidence > 80% found.")

# Select multiple instances to explain
# Example: Select the first two positive and negative examples
num_examples = 2
positive_instance_indices = positive_indices[:num_examples] if len(positive_indices) >= num_examples else positive_indices
negative_instance_indices = negative_indices[:num_examples] if len(negative_indices) >= num_examples else negative_indices

# Initialize LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train,
    feature_names=selected_features,
    class_names=np.unique(Y).astype(str),
    discretize_continuous=True
)

# Function to display and save explanations
def display_and_save_explanations(instance_indices, label):
    for idx in instance_indices:
        instance = X_test[idx]  # Use direct indexing since X_test is a numpy array
        confidence = y_pred_proba[idx][1] if y_pred[idx] == 1 else y_pred_proba[idx][0]
        print(f"Confidence for the {label} instance {idx}: {confidence:.2f}")
        exp = explainer.explain_instance(instance, best_xgb_11.predict_proba, num_features=10)
        
        # Display in the notebook
        exp.show_in_notebook(show_table=True, show_all=False)
        
        # Save the explanation as an image
        fig = exp.as_pyplot_figure()
        plt.tight_layout()
        image_path = os.path.join(output_dir, f'lime_explanation_{label}_best_xgb_11_{idx}.png')
        fig.savefig(image_path, format='png', bbox_inches='tight')
        plt.close(fig)
        print(f'Saved explanation to {image_path}')
        
        # Display the saved image in the notebook
        display(Image(filename=image_path))

# Display and save explanations for positive instances
if len(positive_instance_indices) > 0:
    print("Positive Instances with Confidence > 80%:")
    display_and_save_explanations(positive_instance_indices, 'positive')

# Display and save explanations for negative instances
if len(negative_instance_indices) > 0:
    print("Negative Instances with Confidence > 80%:")
    display_and_save_explanations(negative_instance_indices, 'negative')


