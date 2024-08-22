#!/usr/bin/env python
# coding: utf-8

# ### This notebook contains experiment 12 and 13 of the XGBoost experiments. These had feature selection with Boruta.

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


import sklearn
pd.__version__, np.__version__, sklearn.__version__


# ### Preparing the data

# In[2]:


# Load the imputed dataset after KNN imputation
data = pd.read_csv('your_data')


# In[3]:


data.info()


# In[4]:


data.head(5)


# In[5]:


# Drop the identified columns from the dataset
# Note: may need to use data_stats instead of data_stats_imputed (data stats imputed has the categorical variables imputed)

#removed ethnicity and used ethnic_group instead
#removed APOE4 as per guidance and retained APOE4_alleles
columnns_to_drop_2 = ['eid', 'APOE4', 'ethnicity','LTFU', 'dementia_date', 'dementia_prevalence', 'dementia_incidence', 'Diagnosis_since_baseline', 'Diagnosis_years_since_baseline', 'Years_Group']

data = data.drop(columns=columnns_to_drop_2)


# In[6]:


# Identify the column for the Y variable
target_column = 'dementia_diagnosis'

# Split the data into features (X) and target (Y)
X = data.drop(columns=[target_column])
Y = data[target_column]

# Verify the split
print(f"Features (X) shape: {X.shape}")
print(f"Target (Y) shape: {Y.shape}")


# In[7]:


# Check data types of all columns
data_types = X.dtypes

# Identify categorical variables
categorical_columns = data_types[data_types == 'object'].index.tolist()

print(f'Categorical columns: {categorical_columns}')


# In[8]:


# Convert the 'APOE4_alleles' column to categorical
X['APOE4_alleles'] = X['APOE4_alleles'].astype('category')

# Add 'APOE4_alleles' to the list of categorical columns if not already present
if 'APOE4_alleles' not in categorical_columns:
    categorical_columns.append('APOE4_alleles')


# In[9]:


# One-Hot Encode categorical variables using pd.get_dummies
X_encoded = pd.get_dummies(X, columns=categorical_columns)


# In[10]:


X_encoded.head(10)


# In[11]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, Y, test_size=0.3, random_state=42, stratify=Y)

# Verify the split
print(f"Training features (X_train) shape: {X_train.shape}")
print(f"Testing features (X_test) shape: {X_test.shape}")
print(f"Training target (y_train) shape: {y_train.shape}")
print(f"Testing target (y_test) shape: {y_test.shape}")


# ### Feature selection 

# ### Do feature selection with Boruta

# In[12]:


from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, f1_score, confusion_matrix

# Prepare data for Boruta
X_train_values = X_train.values
X_test_values = X_test.values

# Apply Boruta for feature selection on the training data
rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=7, n_estimators=100, random_state=42)
boruta_selector = BorutaPy(rf, n_estimators='auto', perc=100, max_iter=50, random_state=42)
boruta_selector.fit(X_train.values, y_train.values)

# Select the features
X_train_selected = boruta_selector.transform(X_train.values)
X_test_selected = boruta_selector.transform(X_test.values)
selected_feature_names = np.array(X_encoded.columns)[boruta_selector.support_]
print("Selected features:", selected_feature_names)
print("Number of selected features:", X_train_selected.shape[1])


# In[13]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier

# Fit the RandomForestClassifier again on the selected features to get feature importances
rf_selected = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=7, n_estimators=100, random_state=42)
rf_selected.fit(X_train_selected, y_train)
feature_importances = rf_selected.feature_importances_

# Create a DataFrame with feature names and their importances
importance_df = pd.DataFrame({
    'Feature': selected_feature_names,
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


# In[14]:


# Plot the top features with improved visualization
plt.figure(figsize=(12, 10))
bars = plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Top Features Selected by Boruta')
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Add data labels
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.001, bar.get_y() + bar.get_height() / 2, 
             f'{width:.3f}', ha='left', va='center')
plt.show()

# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'Top Features Selected by Boruta_2.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory

# Display the top features with their importances
importance_df


# In[15]:


import matplotlib.pyplot as plt

# Assuming importance_df is already defined and contains 'Feature' and 'Importance' columns

# Plot the top features with improved visualization
plt.figure(figsize=(14, 12))

# Create the bar plot
bars = plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')

# Set the labels and title
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Top Features Selected by Boruta')
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Add data labels
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.001, bar.get_y() + bar.get_height() / 2, 
             f'{width:.3f}', ha='left', va='center', fontsize=8)

# Adjust font size for feature names and rotate y-axis labels if needed
plt.yticks(fontsize=8)

# Adjust subplots to add more space on the left
plt.subplots_adjust(left=0.3)

plt.show()

# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'Top Features Selected by Boruta_3.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory

# Display the top features with their importances
importance_df


# In[16]:


import matplotlib.pyplot as plt

# Assuming importance_df is already defined and contains 'Feature' and 'Importance' columns

# Plot the top features with improved visualization
plt.figure(figsize=(14, 20))  # Increase the height of the figure

# Create the bar plot with adjusted bar width
bars = plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue', edgecolor='black')

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
plt.subplots_adjust(left=0.35)  # Increase the left margin

# Show and save the plot
plt.show()

# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'Top Features Selected by Boruta_3.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory

# Display the top features with their importances
importance_df


# In[42]:


import matplotlib.pyplot as plt

# Assuming importance_df is already defined and contains 'Feature' and 'Importance' columns

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


# In[17]:


# Standardize the data after splitting
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_selected)
X_test = scaler.transform(X_test_selected)


# In[18]:


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


# ### Experiment 12: XG Boost with no oversampling

# In[19]:


import xgboost

xgboost.__version__


# In[20]:


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
print(f'Nested CV ROC AUC Model 1: {nested_cv_results["test_roc_auc"].mean():.4f} ± {nested_cv_results["test_roc_auc"].std():.4f}')
print(f'Nested CV Recall Model 1: {nested_cv_results["test_recall"].mean():.4f} ± {nested_cv_results["test_recall"].std():.4f}')
print(f'Nested CV Precision Model 1: {nested_cv_results["test_precision"].mean():.4f} ± {nested_cv_results["test_precision"].std():.4f}')
print(f'Nested CV F1 Model 1: {nested_cv_results["test_f1"].mean():.4f} ± {nested_cv_results["test_f1"].std():.4f}')
print(f'Nested CV Specificity Model 1: {nested_cv_results["test_specificity"].mean():.4f} ± {nested_cv_results["test_specificity"].std():.4f}')
print(f'Nested CV Accuracy Model 1: {nested_cv_results["test_accuracy"].mean():.4f} ± {nested_cv_results["test_accuracy"].std():.4f}')

# Fit the model
random_search_5.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_test, y_test)], verbose=10)

# Save the final model
def save_checkpoint(model, filename="checkpoint.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(model, f)
save_checkpoint(random_search_5, "final_model.pkl")

# Best hyperparameters
best_params_5 = random_search_5.best_params_
print(f'Best hyperparameters Model 1: {best_params_5}')

# Test set evaluation
best_xgb_5 = random_search_5.best_estimator_
y_pred_proba = best_xgb_5.predict_proba(X_test)[:, 1]
test_roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f'Test set ROC AUC Model 1: {test_roc_auc:.4f}')

# Additional test set metrics
y_pred = best_xgb_5.predict(X_test)
test_recall = recall_score(y_test, y_pred)
test_precision = precision_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred)
test_specificity = specificity_score(y_test, y_pred)

print(f'Test set Recall Model 1: {test_recall:.4f}')
print(f'Test set Precision Model 1: {test_precision:.4f}')
print(f'Test set F1 Model 1: {test_f1:.4f}')
print(f'Test set Specificity Model 1: {test_specificity:.4f}')


# In[21]:


from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay, recall_score, precision_score, f1_score, roc_curve, roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt

# make predictions using the best model from cross validation in test dataset
y_pred= best_xgb_5.predict(X_test)

# Classification report
print("Classification Report Model 1:")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("Confusion Matrix Model 1:")
ConfusionMatrixDisplay.from_estimator(best_xgb_5, X_test, y_test, cmap=plt.cm.Blues)
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
y_pred_proba = best_xgb_5.predict_proba(X_test)[:, 1]
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


# In[22]:


import matplotlib.pyplot as plt

# Extract feature names from selected_feature_names
feature_names = list(selected_feature_names)  # Convert Index to list

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
print("Top features based on gain:")
print(importance_gain_df.head(10))

# Plot top features by gain
plt.figure(figsize=(10, 6))
plt.barh(importance_gain_df['Feature'].head(10), importance_gain_df['Gain'].head(10))
plt.xlabel('Gain')
plt.ylabel('Feature')
plt.title('Top 10 Features by Gain')
plt.gca().invert_yaxis()
plt.show()

# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'Top 10 Features by Gain_model1.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory


# In[23]:


import shap

shap.__version__


# In[27]:


import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

# Create a SHAP explainer for your model
explainer = shap.Explainer(best_xgb_5)

# Calculate SHAP values for the training set
shap_values = explainer(X_train)

# SHAP summary plot (bar)
shap.summary_plot(shap_values, features=X_train, feature_names=selected_feature_names, plot_type="bar")

# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'SHAP summary plot_bar_model1.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory

# SHAP detailed summary plot
shap.summary_plot(shap_values, features=X_train, feature_names=selected_feature_names)

# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'SHAP summary plot_model1.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory


# In[28]:


# Create a SHAP explainer for your model
explainer = shap.Explainer(best_xgb_5)

# Calculate SHAP values for the test set
shap_values = explainer(X_test)

# SHAP summary plot (bar)
shap.summary_plot(shap_values, features=X_test, feature_names=selected_feature_names, plot_type="bar")
output_path = os.path.join(output_dir, 'shap_values_bar_model5_test.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory

# SHAP detailed summary plot
shap.summary_plot(shap_values, features=X_test, feature_names=selected_feature_names)
output_path = os.path.join(output_dir, 'shap_values_model5_test.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory


# In[29]:


# Create a SHAP explainer for your model
explainer = shap.Explainer(best_xgb_5.predict, X_test)

# Calculate SHAP values for the test set
shap_values = explainer(X_test)

# SHAP summary plot (bar)
shap.summary_plot(shap_values, features=X_test, feature_names=selected_feature_names, plot_type="bar")

# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'SHAP_summary_plot_bar_model1.1.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory

# SHAP detailed summary plot
shap.summary_plot(shap_values, features=X_test, feature_names=selected_feature_names)

# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'SHAP_summary_plot_model1.2.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory


# In[30]:


# pip install lime


# ### Looking at LIME

# In[31]:


import lime
import lime.lime_tabular


# In[32]:


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
    X_train,  # No need to use .values since it's already a numpy array
    feature_names=selected_feature_names,
    class_names=np.unique(Y).astype(str),
    discretize_continuous=True
)

# Function to display and save explanations
def display_and_save_explanations(instance_indices, label):
    for idx in instance_indices:
        instance = X_test[idx]  # Access the instance directly since X_test is a numpy array
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


# ### Experiment 13: same as experiment 12 but with oversampling the minority class

# In[33]:


import imblearn

imblearn.__version__


# In[34]:


import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, make_scorer, confusion_matrix, accuracy_score
import pickle
from imblearn.over_sampling import RandomOverSampler

# Custom scoring function for specificity
def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

# Apply RandomOverSampler to oversample the minority class
ros = RandomOverSampler(random_state=42)
X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)

# Define a parameter grid with more granular alpha and lambda values
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
    'specificity': make_scorer(specificity_score),
    'accuracy': 'accuracy'
}

# Randomized search
random_search_4 = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_dist, cv=inner_cv, n_iter=10, scoring='roc_auc', n_jobs=2, verbose=2, random_state=42)

# Nested cross-validation
nested_cv_results = cross_validate(random_search_4, X_train_ros, y_train_ros, cv=outer_cv, scoring=scoring, n_jobs=2, verbose=1, return_train_score=True)

# Print the mean and standard deviation of the nested cross-validation scores
print(f'Nested CV ROC AUC Model 2: {nested_cv_results["test_roc_auc"].mean():.4f} ± {nested_cv_results["test_roc_auc"].std():.4f}')
print(f'Nested CV Recall Model 2: {nested_cv_results["test_recall"].mean():.4f} ± {nested_cv_results["test_recall"].std():.4f}')
print(f'Nested CV Precision Model 2: {nested_cv_results["test_precision"].mean():.4f} ± {nested_cv_results["test_precision"].std():.4f}')
print(f'Nested CV F1 Model 2: {nested_cv_results["test_f1"].mean():.4f} ± {nested_cv_results["test_f1"].std():.4f}')
print(f'Nested CV Specificity Model 2: {nested_cv_results["test_specificity"].mean():.4f} ± {nested_cv_results["test_specificity"].std():.4f}')
print(f'Nested CV Accuracy Model 2: {nested_cv_results["test_accuracy"].mean():.4f} ± {nested_cv_results["test_accuracy"].std():.4f}')

# Fit the model
random_search_4.fit(X_train_ros, y_train_ros, early_stopping_rounds=10, eval_set=[(X_test, y_test)], verbose=10)

# Save the final model
def save_checkpoint(model, filename="checkpoint.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(model, f)
save_checkpoint(random_search_4, "final_model.pkl")

# Best hyperparameters
best_params_4 = random_search_4.best_params_
print(f'Best hyperparameters Model 2: {best_params_4}')

# Test set evaluation
best_xgb_4 = random_search_4.best_estimator_

y_pred_proba = best_xgb_4.predict_proba(X_test)[:, 1]
test_roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f'Test set ROC AUC Model 2: {test_roc_auc:.4f}')

# Additional test set metrics
y_pred = best_xgb_4.predict(X_test)
test_recall = recall_score(y_test, y_pred)
test_precision = precision_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred)
test_specificity = specificity_score(y_test, y_pred)

print(f'Test set Recall Model 2: {test_recall:.4f}')
print(f'Test set Precision Model 2: {test_precision:.4f}')
print(f'Test set F1 Model 2: {test_f1:.4f}')
print(f'Test set Specificity Model 2: {test_specificity:.4f}')


# In[35]:


y_pred= best_xgb_4.predict(X_test)

# Classification report
print("Classification Report Model 2:")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("Confusion Matrix Model 2:")
ConfusionMatrixDisplay.from_estimator(best_xgb_4, X_test, y_test, cmap=plt.cm.Blues)
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
y_pred_proba = best_xgb_4.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC AUC Model 2: {roc_auc:.2f}")

# Plot ROC Curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
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


# In[36]:


import matplotlib.pyplot as plt

# Extract feature names from selected_feature_names
feature_names = list(selected_feature_names)  # Convert Index to list

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


# In[37]:


import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create a SHAP explainer for your model
explainer = shap.Explainer(best_xgb_4)

# Calculate SHAP values for the training set
shap_values = explainer(X_train)

# SHAP summary plot (bar)
shap.summary_plot(shap_values, features=X_train, feature_names=selected_feature_names, plot_type="bar")

# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'SHAP summary plot_bar_model2.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory


# SHAP detailed summary plot
shap.summary_plot(shap_values, features=X_train, feature_names=selected_feature_names)

# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'SHAP summary plot_model2.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory


# In[38]:


# Create a SHAP explainer for your model
explainer = shap.Explainer(best_xgb_4)

# Calculate SHAP values for the test set
shap_values = explainer(X_test)

# SHAP summary plot (bar)
shap.summary_plot(shap_values, features=X_test, feature_names=selected_feature_names, plot_type="bar")
output_path = os.path.join(output_dir, 'shap_values_bar_model4_test.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory

# SHAP detailed summary plot
shap.summary_plot(shap_values, features=X_test, feature_names=selected_feature_names)
output_path = os.path.join(output_dir, 'shap_values_model4_test.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory


# In[39]:


# Create a SHAP explainer for your model
explainer = shap.Explainer(best_xgb_4.predict, X_test)

# Calculate SHAP values for the test set
shap_values = explainer(X_test)

# SHAP summary plot (bar)
shap.summary_plot(shap_values, features=X_test, feature_names=selected_feature_names, plot_type="bar")

# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'SHAP_summary_plot_bar_model2.1.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory

# SHAP detailed summary plot
shap.summary_plot(shap_values, features=X_test, feature_names=selected_feature_names)

# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'SHAP_summary_plot_model2.2.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory


# ### Looking at LIME

# In[40]:


import lime
import lime.lime_tabular


# In[41]:


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
    X_train,  # No need to use .values since it's already a numpy array
    feature_names=selected_feature_names,
    class_names=np.unique(Y).astype(str),
    discretize_continuous=True
)

# Function to display and save explanations
def display_and_save_explanations(instance_indices, label):
    for idx in instance_indices:
        instance = X_test[idx]  # Access the instance directly since X_test is a numpy array
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


# In[ ]:





# In[ ]:





# In[ ]:




