#!/usr/bin/env python
# coding: utf-8

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


# In[3]:


# Load the imputed dataset
data = pd.read_csv('/data/home/ha23130/Scripts_final/data_stats_imputed_knn.csv')


# ### Preparing the data

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


# ### Random Forest classifiers

# ### Experiment 1: settings inpired in Musto el al (2023)

# In[13]:


import pickle


# In[ ]:


import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split, StratifiedKFold

# Calculate sqrt(n_features)
sqrt_n_features = int(np.sqrt(X_train.shape[1]))

# Define the parameter grid
param_grid = {
    'n_estimators': [1000],#same number as Musto et al
    'max_features': list(range(1,21,5)), # Including sqrt(n_features). slight tweak to parameters for computation reasons
    'min_samples_leaf': [10, 20, 30, 40, 50],  # Reasonable range for min_samples_leaf
    'class_weight': ['balanced']  # Adding class_weight to handle class imbalance
}

# Set up the Random Forest classifier
rf = RandomForestClassifier(random_state=42)

# Inner stratified 5-fold cross-validation
inner_cv = StratifiedKFold(n_splits=5)

# Outer stratified 5-fold cross-validation
outer_cv = StratifiedKFold(n_splits=5)

# Set up GridSearchCV with verbose output
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=inner_cv, scoring='accuracy', n_jobs=-1, verbose=2)

# Function to save progress
def save_checkpoint(grid_search, filename="grid_search_checkpoint.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(grid_search, f)

# Nested cross-validation with 5 outer folds
print("Starting nested cross-validation...")
nested_cv_scores = cross_val_score(grid_search, X_train, y_train, cv=outer_cv, scoring='accuracy', n_jobs=-1, verbose=1)

# Save checkpoint after cross-validation
save_checkpoint(grid_search)

# Print the mean and standard deviation of the nested cross-validation scores
print(f'Nested CV Accuracy: {nested_cv_scores.mean():.4f} ± {nested_cv_scores.std():.4f}')

# Fit the model with the entire training set
print("Fitting the best model...")
grid_search.fit(X_train, y_train)

# Save final model
save_checkpoint(grid_search, "final_model.pkl")

# Get the best hyperparameters
best_params = grid_search.best_params_
print(f'Best hyperparameters: {best_params}')

# Evaluate the model on the test set
best_rf = grid_search.best_estimator_
test_accuracy = best_rf.score(X_test, y_test)
print(f'Test set accuracy: {test_accuracy:.4f}')


# 'max_features': list(range(5,10,15)) + [sqrt_n_features], 

# In[ ]:


from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay, recall_score, precision_score, f1_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# make predictions using the best model from cross validation in test dataset
y_pred= best_rf.predict(X_test)

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("Confusion Matrix:")
ConfusionMatrixDisplay.from_estimator(best_rf, X_test, y_test, cmap=plt.cm.Blues)
plt.show()

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Recall score (sensitivity)
recall = recall_score(y_test, y_pred)
print(f"Recall (Sensitivity): {recall:.2f}")

# Specificity
specificity = recall_score(y_test, y_pred, pos_label=0)
print(f"Specificity: {specificity:.2f}")

# Precision
precision = precision_score(y_test, y_pred)
print(f"Precision: {precision:.2f}")

# F1 score
f1 = f1_score(y_test, y_pred)
print(f"F1 Score: {f1:.2f}")

# ROC AUC
y_pred_proba = best_rf.predict_proba(X_test)[:, 1]
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
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()


# In[ ]:





# In[ ]:




