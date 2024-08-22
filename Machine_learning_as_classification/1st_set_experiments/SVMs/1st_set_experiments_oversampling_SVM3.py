#!/usr/bin/env python
# coding: utf-8

# ### This is the work described in section 5.4.1.3.1 (for SVMs with oversampling). This was run via the shell (not via jupyter notebook)

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


# Load the imputed dataset
data = pd.read_csv('include_your_directory')


# In[3]:


data.info()


# In[4]:


data.head(5)


# In[5]:


# Drop the identified columns from the dataset

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


# ### Feature selection 

# ### Do feature selection with Boruta

# In[11]:


from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, f1_score, confusion_matrix

# Prepare data for Boruta
X = X_encoded.values

# Apply Boruta for feature selection
# Setting n_estimators to a fixed number and limiting max_iter
rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=7, n_estimators=100, random_state=42)
boruta_selector = BorutaPy(rf, n_estimators=100, perc=100, max_iter=50, random_state=42)
boruta_selector.fit(X, Y)

# Select the features
selected_features = boruta_selector.transform(X)
selected_feature_names = X_encoded.columns[boruta_selector.support_]
print("Selected features:", selected_feature_names)
print("Number of selected features:", selected_features.shape[1])


# In[1]:


### better visualised the features in a graph


# In[12]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier

# Get the selected feature indices and names
selected_indices = np.where(boruta_selector.support_)[0]
selected_feature_names = X_encoded.columns[selected_indices]

# Get feature importances from the RandomForest used in Boruta
rf.fit(X, Y)
feature_importances = rf.feature_importances_

# Create a DataFrame with feature names and their importances
importance_df = pd.DataFrame({
    'Feature': X_encoded.columns,
    'Importance': feature_importances
})

# Filter to get only the selected features
importance_df = importance_df.loc[selected_indices]


# Sort the DataFrame by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Create the output directory if it doesn't exist
output_dir = 'ML_work_v15_SVM_feature_reduction_boruta_oversampling'
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


# In[13]:


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


# In[14]:


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


# In[15]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(selected_features, Y, test_size=0.3, random_state=42, stratify=Y)

# Verify the split
print(f"Training features (X_train) shape: {X_train.shape}")
print(f"Testing features (X_test) shape: {X_test.shape}")
print(f"Training target (y_train) shape: {y_train.shape}")
print(f"Testing target (y_test) shape: {y_test.shape}")


# In[16]:


# Standardize the data after splitting
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[17]:


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


# In[18]:


from imblearn.over_sampling import RandomOverSampler

# Oversample the minority class
ros = RandomOverSampler(random_state=42)
X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)


# ###  SVM 3: with oversampling the minority class

# In[21]:


import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, f1_score, confusion_matrix, make_scorer

# Custom scoring function for specificity
def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

# Define parameter grid for SVM
param_grid = {'C': [50, 10, 1.0, 0.1, 0.01], 
              'gamma': ['scale'],
              'kernel': ['rbf']} 

# Define SVM model
svm_model_2 = SVC(probability=True)

# Stratified K-Fold cross-validation
cv = StratifiedKFold(n_splits=5)

# Define custom scoring functions
scoring = {
    'roc_auc': 'roc_auc',
    'recall': 'recall',
    'precision': 'precision',
    'f1': 'f1',
    'specificity': make_scorer(specificity_score),
    'accuracy': 'accuracy'
}

# Grid search
grid_search = GridSearchCV(estimator=svm_model_2, param_grid=param_grid, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=2)

# Fit the model
grid_search.fit(X_train_ros, y_train_ros)

# Best hyperparameters
best_params_2 = grid_search.best_params_
print(f'Best hyperparameters Model 2: {best_params_2}')

# Test set evaluation
best_svm_2 = grid_search.best_estimator_
y_pred_proba = best_svm_2.predict_proba(X_test)[:, 1]
test_roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f'Test set ROC AUC Model 2: {test_roc_auc:.4f}')


# Additional test set metrics
y_pred = best_svm_2.predict(X_test)
test_recall = recall_score(y_test, y_pred, zero_division=1)
test_precision = precision_score(y_test, y_pred, zero_division=1)
test_f1 = f1_score(y_test, y_pred, zero_division=1)
test_specificity = specificity_score(y_test, y_pred)
test_accuracy = accuracy_score(y_test, y_pred)

print(f'Test set Recall Model 2: {test_recall:.4f}')
print(f'Test set Precision Model 2: {test_precision:.4f}')
print(f'Test set F1 Model 2: {test_f1:.4f}')
print(f'Test set Specificity Model 2: {test_specificity:.4f}')
print(f'Test set Accuracy Model 2: {test_accuracy:.4f}')


# In[ ]:


from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay, recall_score, precision_score, f1_score, roc_curve, roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("Confusion Matrix:")
ConfusionMatrixDisplay.from_estimator(best_svm_2, X_test, y_test, cmap=plt.cm.Blues)
plt.show()

# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'Confusion Matrix_SVM_2.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory

# ROC AUC
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
plt.title('Receiver Operating Characteristic_svm_2')
plt.legend(loc='lower right')
plt.show()

# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'Receiver Operating Characteristic_SVM_2.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory


# Precision-Recall Curve
precisions, recalls, _ = precision_recall_curve(y_test, y_pred)
plt.figure()
plt.plot(recalls, precisions, color='blue', lw=2, label=f'PR curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve_2')
plt.legend(loc='lower left')
plt.show()

# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'Precision-Recall Curve_SVM_2.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory


# In[2]:


### interpretability with SHAP


# In[ ]:


import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Create a SHAP KernelExplainer for your model
explainer = shap.KernelExplainer(best_svm_2.predict, X_train)

# Calculate SHAP values for the sampled test set
shap_values = explainer.shap_values(X_train)

# SHAP summary plot (bar)
shap.summary_plot(shap_values, features=X_train, feature_names=selected_feature_names, plot_type="bar")

# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'SHAP_summary_plot_bar_SVM_2.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory

# SHAP detailed summary plot
shap.summary_plot(shap_values, features=X_test_sampled, feature_names=selected_feature_names)

# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'SHAP_summary_plot_SVM_2.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory

