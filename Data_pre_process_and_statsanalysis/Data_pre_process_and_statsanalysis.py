#!/usr/bin/env python
# coding: utf-8

# ## Data pre_processing script (section 5.1 and 5.2)

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


# In[5]:


# Load the datasets
proteomics_data = pd.read_csv('proteomics_data_cleaned.csv') 

# Read the CSV file, specifying the correct index column
metabolomics_data = pd.read_csv('nmr_qc_derived.csv', index_col=0)


# In[6]:


proteomics = proteomics_data 
metabolomics = metabolomics_data


# In[7]:


# number of unique participants in each dataset
proteomics_participants = proteomics['eid'].nunique()
metabolomics_participants = metabolomics['eid'].nunique()

print(f"Number of unique participants in proteomics data: {proteomics_participants}")
print(f"Number of unique participants in metabolomics data: {metabolomics_participants}")


# ### Understand the data

# In[8]:


proteomics.info()


# In[9]:


proteomics.head(2)


# In[11]:


metabolomics.info()


# In[12]:


metabolomics.head(2)


# In[14]:


# understand the visit index count (baseline and follow up)
metabolomics['visit_index'].value_counts()


# In[15]:


# Count the number of entries per participant (eid)
metabolomics['eid'].value_counts()


# In[16]:


# Display the data for a specific participant ID
specific_participant_id = 5036664
print(metabolomics[metabolomics['eid'] == specific_participant_id])


# In[17]:


#Include only metabolomics baseline values
# Step 1: Filter entries with visit_index 0
metabolomics= metabolomics[metabolomics['visit_index'] == 0]

# Reset the index for the filtered DataFrame
metabolomics.reset_index(drop=True, inplace=True)


# In[18]:


metabolomics.head(2)


# In[19]:


metabolomics['visit_index'].value_counts()


# In[20]:


metabolomics_data_1 = metabolomics


# ### Merge datasets

# In[21]:


# Join based on proteomics data, including only those participants that also have metabolomics data
merged_on_proteomics_inner = pd.merge(proteomics, metabolomics_data_1, on='eid', how='inner')
participants_after_proteomics_inner_join = merged_on_proteomics_inner['eid'].nunique()

print(f"Number of participants after inner joining on proteomics data: {participants_after_proteomics_inner_join}")


# In[22]:


merged_on_proteomics_inner.head(5)


# In[23]:


# Dementia cases before prevelance removal
dementia_cases = merged_on_proteomics_inner[merged_on_proteomics_inner['dementia_diagnosis'] == 1]

dementia_cases.info()


# In[24]:


merged_on_proteomics_inner.dementia_diagnosis.value_counts()


# ### Remove prevalence

# In[25]:


dataset = merged_on_proteomics_inner


# In[26]:


dataset.info()


# In[27]:


dataset['eid'].nunique()


# In[28]:


# Remove rows where 'dementia_prevalence' is 1.0
dataset= dataset[dataset['dementia_prevalence'] != 1.0]

#check rows and columns
len(dataset)


# In[29]:


# Dementia cases 
dataset.dementia_diagnosis.value_counts()


# In[30]:


#print("Column names:", dataset.columns.tolist())


# ### Number of proteomics and Metabolomics

# In[31]:


# Get the index of the starting columns for proteomics and metabolomics
proteomics_start_col = 'AAMDC'
metabolomics_start_col = 'bOHbutyrate'

# Get the list of column names
columns = dataset.columns.tolist()

# Find the start index of proteomics and metabolomics columns
proteomics_start_idx = columns.index(proteomics_start_col)
metabolomics_start_idx = columns.index(metabolomics_start_col)

# Proteomics columns are from proteomics_start_idx to one before metabolomics_start_idx
proteomics_columns = columns[proteomics_start_idx:metabolomics_start_idx]

# Metabolomics columns are from metabolomics_start_idx to the end
metabolomics_columns = columns[metabolomics_start_idx:]

# Count the number of proteomics and metabolomics columns
num_proteomics = len(proteomics_columns)
num_metabolomics = len(metabolomics_columns)

print(f"Number of proteomics variables: {num_proteomics}")
print(f"Number of metabolomics variables: {num_metabolomics}")


# ### Dementia cases before removing missing data

# In[32]:


# Now filter the DataFrame to include only cases with dementia diagnosis
dementia_cases = dataset[dataset['dementia_diagnosis'] == 1]


# In[33]:


print(dementia_cases.info())


# ### Remove columns where more than 25% of data is missing 

# In[34]:


# Set the threshold for missing values (21%)
threshold = 0.25

# Calculate the percentage of missing values for each column
missing_percentage_columns = dataset.isnull().mean()

# Identify columns with more than 25% missing data
columns_to_drop = missing_percentage_columns[missing_percentage_columns > threshold].index

# Define the columns to exclude from dropping
columns_to_exclude = ['eid', 'gender', 'age_at_baseline', 'ethnicity', 'ethnic_group','years_education', 'BMI_levels', 'LTFU', 'dementia_diagnosis', 'dementia_date', 'dementia_prevalence', 'dementia_incidence', 'Diagnosis_since_baseline', 'Diagnosis_years_since_baseline']

# Exclude the defined columns from being dropped
columns_to_drop = [col for col in columns_to_drop if col not in columns_to_exclude]

# Calculate the percentage of missing data for the columns that were removed
missing_percentage_dropped_columns = missing_percentage_columns[columns_to_drop]

# Display the percentage of missing data for the removed columns
print(f"Percentage of missing data for the removed columns:")
for column, missing_percentage in missing_percentage_dropped_columns.items():
    print(f"{column}: {missing_percentage:.2%}")

# Drop the identified columns from the dataset
dataset_nomissing = dataset.drop(columns=columns_to_drop)

# Reset the index for the filtered DataFrame if needed (optional)
dataset_nomissing.reset_index(drop=True, inplace=True)

# Display the columns that were dropped
print(f"Columns dropped due to more than 25% missing data: {columns_to_drop}")

# Verify the remaining columns
print("Remaining columns in the dataset after dropping:")
print(dataset_nomissing.columns)

# Compare the number of columns before and after filtering
initial_column_count = len(dataset.columns)
filtered_column_count = len(dataset_nomissing.columns)
print(f"Initial number of columns: {initial_column_count}")
print(f"Number of columns after filtering: {filtered_column_count}")


# In[35]:


# Compare the number of columns before and after filtering
initial_column_count = len(dataset.columns)
filtered_column_count = len(dataset_nomissing.columns)
print(f"Initial number of columns: {initial_column_count}")
print(f"Number of columns after filtering: {filtered_column_count}")


# In[36]:


dataset_nomissing.info()


# In[37]:


# Filter the DataFrame to include only cases with dementia diagnosis
dementia_cases_1 = dataset_nomissing[dataset_nomissing['dementia_diagnosis'] == 1]
dementia_cases_1.info()


# In[39]:


# Q&A to test if removal of data was done correctly

# Verify that no column in the filtered dataset has more than 20% missing values
max_missing_percentage_col = dataset_nomissing.isnull().mean().max()
assert max_missing_percentage_col >= 0.25, f"Some columns have more than 25% missing values: {max_missing_percentage_col}"

# Compare the number of columns before and after filtering
initial_column_count = len(dataset.columns)
filtered_column_count = len(dataset_nomissing.columns)
print(f"Initial number of columns: {initial_column_count}")
print(f"Number of columns after filtering: {filtered_column_count}")

# Display some columns to manually inspect
print("Sample columns from the filtered dataset:")
print(dataset_nomissing.sample(5, axis=1))  # Display 5 random columns

# Display the missing value percentage for the displayed columns for confirmation
sample_missing_percentage_col = dataset_nomissing.sample(5, axis=1).isnull().mean()
print("Missing value percentage for the sample columns:")
print(sample_missing_percentage_col)


# In[40]:


# Calculate the percentage of missing values for each column in the filtered dataset
missing_percentage_columns = dataset_nomissing.isnull().mean()

# Find the maximum and minimum percentage of missing data in the filtered columns
max_missing_percentage_col = missing_percentage_columns.max()
min_missing_percentage_col = missing_percentage_columns.min()

# Display the results
print(f"Maximum percentage of missing data in filtered columns: {max_missing_percentage_col:.2%}")
print(f"Minimum percentage of missing data in filtered columns: {min_missing_percentage_col:.2%}")


# In[41]:


# Set a threshold to investigate high missing percentage columns
high_missing_threshold = 0.25

# Identify columns with more than the specified threshold of missing data
high_missing_columns = missing_percentage_columns[missing_percentage_columns > high_missing_threshold]

# Display the columns with high missing data percentages
print(f"Columns with more than {high_missing_threshold*100}% missing data:")
print(high_missing_columns)


# ### Remove participants where 50% or more data is missing

# In[42]:


import pandas as pd
import matplotlib.pyplot as plt
import os

# Set the threshold for missing values (20%)
threshold = 0.5

# Calculate the percentage of missing values for each row
missing_percentage = dataset_nomissing.isnull().mean(axis=1)

# Identify the rows with more than 20% missing data
rows_to_remove = missing_percentage[missing_percentage > threshold]

# Display the EID and the percentage of missing data for the removed rows
print("EIDs and percentage of missing data for the removed rows:")
for idx in rows_to_remove.index:
    eid = dataset_nomissing.iloc[idx]['eid']
    missing_percentage_value = rows_to_remove[idx]
    print(f"EID: {eid}, Missing data: {missing_percentage_value:.2%}")

    
# Create the output directory if it doesn't exist
output_dir = 'output_data_analysis'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Collect EIDs and their missing data percentages
removed_data = pd.DataFrame({
    'EID': dataset_nomissing.loc[rows_to_remove.index, 'eid'],
    'Missing_Percentage': rows_to_remove.values * 100  # Convert to percentage
})

# Plot the data
plt.figure(figsize=(10, 6))

# Plot individual points
plt.scatter(range(len(removed_data)), removed_data['Missing_Percentage'], s=10)
plt.xlabel('Index')
plt.ylabel('Percentage of Missing Data')
plt.title('Percentage of Missing Data for Removed EIDs')


# Add horizontal lines for min, median, mean, and max
min_value = removed_data['Missing_Percentage'].min()
median_value = removed_data['Missing_Percentage'].median()
mean_value = removed_data['Missing_Percentage'].mean()
max_value = removed_data['Missing_Percentage'].max()

plt.axhline(min_value, color='green', linestyle='dashed', linewidth=1, label=f'Min: {min_value:.2f}%')
plt.axhline(median_value, color='blue', linestyle='dashed', linewidth=1, label=f'Median: {median_value:.2f}%')
plt.axhline(mean_value, color='orange', linestyle='dashed', linewidth=1, label=f'Mean: {mean_value:.2f}%')
plt.axhline(max_value, color='red', linestyle='dashed', linewidth=1, label=f'Max: {max_value:.2f}%')

plt.legend()
plt.tight_layout()
plt.show()
# Save the plot as a file in the output directory
output_path = os.path.join(output_dir, 'Percentage of Missing Data for Removed EIDs.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory

# Filter the DataFrame to include only rows with <= 20% missing values
dataset_nomissing_filtered = dataset_nomissing[missing_percentage <= threshold].copy()

# Reset the index for the filtered DataFrame
dataset_nomissing_filtered.reset_index(drop=True, inplace=True)

# Verify the number of remaining rows
print("Number of remaining rows after filtering:")
print(len(dataset_nomissing_filtered))

# Display the filtered dataset
print("Filtered metabolomics data:")
print(dataset_nomissing_filtered)


# In[43]:


# Verify that no row in the filtered dataset has more than 50% missing values
max_missing_percentage = dataset_nomissing_filtered.isnull().mean(axis=1).max()
assert max_missing_percentage <= 0.5, f"Some rows have more than 50% missing values: {max_missing_percentage}"

# Compare the number of rows before and after filtering
initial_row_count = len(dataset)
clean_variables_row_count = len(dataset_nomissing)
filtered_row_count = len(dataset_nomissing_filtered)
print(f"Initial number of rows: {initial_row_count}")
print(f"Rows after variables removal: {clean_variables_row_count}")
print(f"Number of rows after filtering: {filtered_row_count}")

# Display some rows to manually inspect
print("Sample rows from the filtered dataset:")
print(dataset_nomissing_filtered.sample(5))  # Display 5 random rows

# Display the missing value percentage for the displayed rows for confirmation
sample_missing_percentage = dataset_nomissing_filtered.sample(5).isnull().mean(axis=1)
print("Missing value percentage for the sample rows:")
print(sample_missing_percentage)


# In[44]:


# Calculate the percentage of missing values for each row in the filtered dataset
missing_percentage_filtered = dataset_nomissing_filtered.isnull().mean(axis=1)

# Find the maximum and minimum percentage of missing data in the filtered rows
max_missing_percentage = missing_percentage_filtered.max()
min_missing_percentage = missing_percentage_filtered.min()

# Display the results
print(f"Maximum percentage of missing data in filtered rows: {max_missing_percentage:.2%}")
print(f"Minimum percentage of missing data in filtered rows: {min_missing_percentage:.2%}")


# In[45]:


# filter the DataFrame to include only cases with dementia diagnosis
dementia_cases = dataset_nomissing_filtered[dataset_nomissing_filtered['dementia_diagnosis'] == 1]
dementia_cases.info()


# In[46]:


dataset_nomissing_filtered.info()


# ### Final number of dementia cases and controls after exclusions applied

# In[47]:


dataset_nomissing_filtered.dementia_diagnosis.value_counts()


# ### Create new variable Years_Group (to categorise cases into thresholds)

# In[48]:


import pandas as pd
import matplotlib.pyplot as plt

# Define the bins and labels
bins = [-float('inf'), 0, 5, 10, float('inf')]
labels = ['Up to baseline', 'Up to 5 years', '5 to 10 years', 'Above 10 years']  # Labels for the plot

# Define numeric codes for each category
numeric_codes = [1, 2, 3, 4]  # Numeric codes for the categories

# Create a dictionary to map numeric codes to bin intervals
bin_dict = {code: bins[i:i+2] for i, code in enumerate(numeric_codes)}

# Add 'Controls' to the dictionary (no specific interval)
bin_dict[0] = 'No specific interval (participants without dementia diagnosis)'

# Categorise the data into bins and assign numeric codes
dataset_nomissing_filtered['Years_Group'] = pd.cut(dataset_nomissing_filtered['Diagnosis_years_since_baseline'], bins=bins, labels=numeric_codes, right=False).astype(float)

# Identify controls (participants with no dementia diagnosis)
controls_mask = dataset_nomissing_filtered['dementia_diagnosis'] == 0
dataset_nomissing_filtered.loc[controls_mask, 'Years_Group'] = 0

# Calculate the value counts and percentages
value_counts = dataset_nomissing_filtered['Years_Group'].value_counts().sort_index()
value_percentages = dataset_nomissing_filtered['Years_Group'].value_counts(normalize=True).sort_index() * 100

# Combine counts and percentages into a DataFrame
distribution_df = pd.DataFrame({'Count': value_counts, 'Percentage': value_percentages})

# Display the counts and percentages, including the number of controls
print(distribution_df)

# Print the bin dictionary
print("\nBin intervals associated with each label:")
for code, interval in bin_dict.items():
    print(f"{code}: {interval}")

# Print key-value pairs for each category
for category, row in distribution_df.iterrows():
    print(f"{category}: Count = {row['Count']}, Percentage = {row['Percentage']:.2f}%")

# Mapping of numeric codes to labels for the plot
code_to_label = {0: 'Controls', 1: 'Up to baseline', 2: 'Up to 5 years', 3: '5 to 10 years', 4: 'Above 10 years'}

# Plot the counts and percentages, excluding 'Controls'
plot_df = distribution_df.drop(index=0)

plt.figure(figsize=(10, 6))
ax = plot_df['Count'].plot(kind='bar', color='skyblue', edgecolor='black')
plt.xlabel('Years Group')
plt.ylabel('Count')
plt.title('Counts of Diagnosis Years Since Baseline Groups (Excluding Controls)')
ax.set_xticklabels([code_to_label[x] for x in plot_df.index], rotation=45, ha='right')

# Add the count and percentage values on top of the bars
for i, p in enumerate(ax.patches):
    count = plot_df['Count'].iloc[i]
    percentage = plot_df['Percentage'].iloc[i]
    ax.annotate(f"{count} ({percentage:.2f}%)", 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')

plt.tight_layout()
plt.show()


output_path = os.path.join(output_dir, 'Counts of Diagnosis Years Since Baseline Groups.png')
plt.savefig(output_path)
plt.close()  # Close the plot to free up memory


# In[55]:


dataset_nomissing_filtered['Years_Group'].value_counts()


# ### Check outliers in the data

# In[67]:


import pandas as pd
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load your data into a DataFrame
data = dataset_nomissing_filtered

# Columns to exclude from the analysis
excluded_columns = ['eid', 'gender', 'age_at_baseline', 'ethnicity', 'ethnic_group','years_education', 'BMI_levels', 'LTFU', 
                    'dementia_diagnosis', 'dementia_date', 'dementia_prevalence', 'dementia_incidence', 
                    'Diagnosis_since_baseline', 'Diagnosis_years_since_baseline', 'Years_Group']

# Strip any whitespace from the column names
data.columns = data.columns.str.strip()

# Verify which excluded columns exist in the DataFrame
existing_excluded_columns = [col for col in excluded_columns if col in data.columns]
print("Existing excluded columns:", existing_excluded_columns)

# Filter only numeric columns
numeric_data = data.select_dtypes(include=[np.number])

# Ensure we only drop columns that exist in the numeric data
columns_to_drop = [col for col in existing_excluded_columns if col in numeric_data.columns]
print("Columns to drop from numeric data:", columns_to_drop)

# Drop the existing excluded columns from numeric data
numeric_data = numeric_data.drop(columns=columns_to_drop)

# Compute the Z-scores of the numeric variables with a lower threshold
z_scores = numeric_data.apply(zscore)

# Define a lower threshold to identify outliers (e.g., Z-score > 2 or < -2)
threshold = 2
outliers = (z_scores > threshold) | (z_scores < -threshold)

# Identify columns with outliers
columns_with_outliers = outliers.any()

# Get the columns with outliers
outlier_columns = columns_with_outliers[columns_with_outliers].index.tolist()
print("Columns with outliers:", outlier_columns)

# Create the output directory if it doesn't exist
output_dir = 'output_plots'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Visualize the distributions of a few columns to understand their spread
sample_columns = numeric_data.columns[:10]  # Adjust this to visualize different columns if needed
for col in sample_columns:
    plt.figure(figsize=(10, 5))
    sns.histplot(numeric_data[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()
    # Save the plot as a file in the output directory
    plt.savefig(os.path.join(output_dir, f'distribution_{col}.png'))
    plt.close()  # Close the plot to free up memory

# Visualize using boxplots for columns with outliers
if len(outlier_columns) > 0:
    numeric_data[outlier_columns].boxplot(figsize=(15, 10))
    plt.xticks(rotation=90)
    plt.show()
    # Save the boxplot as a file in the output directory
    plt.savefig(os.path.join(output_dir, 'boxplot_outliers.png'))
    plt.close()  # Close the plot to free up memory
else:
    print("No outliers detected in the dataset.")


# ### Missing data checks analysis (check for MCAR and MAR assumptions)

# In[78]:


#MCAR/MAR analysis 

# Repeat the sampling and analysis for consistency
for _ in range(5):  # Repeat 5 times
    sampled_columns = np.random.choice(data.columns, size=int(0.2 * len(data.columns)), replace=False)
    sampled_data = data[sampled_columns]

    # Heatmap for missing data
    sns.heatmap(sampled_data.isnull(), cbar=False, cmap="viridis")
    plt.title('Missing Data Heatmap for Another Sample')
    plt.show()

    # Correlation matrix between missing indicators and actual data
    missing_indicators = sampled_data.isnull().astype(int)
    correlation_matrix = missing_indicators.corrwith(sampled_data, axis=0)

    # Plotting the correlations
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix.to_frame(), annot=False, cmap='coolwarm', center=0)
    plt.title('Correlation between Missingness and Observed Data (Another Sample)')
    plt.show()


# In[54]:


dataset_nomissing_filtered.info()


# ### Descriptive statistics

# In[79]:


data_stats = dataset_nomissing_filtered


# In[80]:


data_stats.head()


# In[81]:


# Print the names of the columns
print("Column names:", data_stats.columns.tolist())


# In[82]:


data_stats.APOE4.value_counts()


# In[83]:


data_stats.APOE4_alleles.value_counts()


# In[84]:


data_stats.dementia_diagnosis.value_counts()


# In[85]:


# Check for missing data in the specified variables
#do not concatenate results
variables_of_interest = ['gender', 'age_at_baseline', 'years_education', 'ethnic_group', 'BMI_levels', 'APOE4_alleles', 'APOE4']

missing_data = data_stats[variables_of_interest].isnull().sum()
print(missing_data)


# In[86]:


#do not concatenate results
pd.set_option('display.max_rows', data_stats.shape[0]+1)

missing_data_all = data_stats.isnull().sum()
print(missing_data_all)


# In[87]:


data_stats.describe()


# ### Imputing just the variables for summary stats

# In[88]:


from sklearn.impute import KNNImputer

variables_input = ['years_education','APOE4_alleles', 'APOE4']

# Initialize the KNNImputer
imputer = KNNImputer(n_neighbors=5)

# Perform KNN imputation
data_stats_imputed = data_stats.copy()
data_stats_imputed[variables_input] = imputer.fit_transform(data_stats[variables_input])

# Round the imputed values to the nearest valid category for APOE4_alleles and APOE4
data_stats_imputed['APOE4_alleles'] = data_stats_imputed['APOE4_alleles'].round().astype(int)
data_stats_imputed['APOE4'] = data_stats_imputed['APOE4'].round().astype(int)

# Ensure the imputed values fall within the original categories
data_stats_imputed['APOE4_alleles'] = data_stats_imputed['APOE4_alleles'].clip(lower=0, upper=2)
data_stats_imputed['APOE4'] = data_stats_imputed['APOE4'].clip(lower=0, upper=1)

# Verify the value counts after imputation and rounding
print(data_stats_imputed['APOE4_alleles'].value_counts())
print(data_stats_imputed['APOE4'].value_counts())


# In[89]:


# Check for missing data after imputation
missing_data_after_imputation = data_stats_imputed[variables_of_interest].isnull().sum()
print("\nMissing data in each variable after KNN imputation:")
print(missing_data_after_imputation)


# In[90]:


data_stats_imputed.APOE4_alleles.value_counts()


# In[91]:


data_stats_imputed.APOE4.value_counts()


# In[92]:


variables_of_interest = ['gender', 'age_at_baseline', 'years_education', 'ethnic_group', 'BMI_levels', 'APOE4_alleles', 'APOE4']

# Split the DataFrame based on dementia diagnosis
df_dementia = data_stats_imputed[data_stats_imputed['dementia_diagnosis'] == 1]
df_no_dementia = data_stats_imputed[data_stats_imputed['dementia_diagnosis'] == 0]

# Compute summary statistics for each group
summary_dementia = df_dementia[variables_of_interest].describe(include='all')
summary_no_dementia = df_no_dementia[variables_of_interest].describe(include='all')

# Combine the results into a single DataFrame
summary_combined = pd.concat([summary_dementia, summary_no_dementia], axis=1, keys=['Dementia', 'No Dementia'])

# Display the combined summary statistics
print("Summary Statistics by Dementia Diagnosis:")
print(summary_combined)


# ### Summary descriptive statistics and p-values

# In[93]:


from scipy import stats

# Function to calculate median and IQR
def median_iqr(series):
    median = series.median()
    q25 = series.quantile(0.25)
    q75 = series.quantile(0.75)
    return f"{median} [{q25}-{q75}]"

# Function to format p-values with significance stars
def format_p_value(p):
    if p < 0.001:
        return f"{p:.2e}***"
    elif p < 0.01:
        return f"{p:.2e}**"
    elif p < 0.05:
        return f"{p:.2e}*"
    else:
        return f"{p:.2e} (ns)"

# Summary statistics for continuous variables
summary_continuous = {
    'Variable': [],
    'No Dementia (median [IQR])': [],
    'Dementia (median [IQR])': [],
    'p-value': []
}

for var in ['age_at_baseline', 'years_education']:
    summary_continuous['Variable'].append(var)
    summary_continuous['No Dementia (median [IQR])'].append(median_iqr(df_no_dementia[var]))
    summary_continuous['Dementia (median [IQR])'].append(median_iqr(df_dementia[var]))
    t_stat, p_val = stats.ttest_ind(df_no_dementia[var], df_dementia[var], nan_policy='omit')
    summary_continuous['p-value'].append(format_p_value(p_val))

summary_continuous_df = pd.DataFrame(summary_continuous)

# Summary statistics for categorical variables
summary_categorical = []

for var in ['gender', 'ethnic_group', 'BMI_levels', 'APOE4_alleles', 'APOE4']:
    no_dementia_counts = df_no_dementia[var].value_counts().sort_index()
    dementia_counts = df_dementia[var].value_counts().sort_index()
    no_dementia_perc = (no_dementia_counts / len(df_no_dementia) * 100).round(2)
    dementia_perc = (dementia_counts / len(df_dementia) * 100).round(2)
    
    chi2, p_val = stats.chi2_contingency(pd.concat([no_dementia_counts, dementia_counts], axis=1, keys=['No Dementia', 'Dementia']).fillna(0).astype(int))[0:2]
    
    for idx in no_dementia_counts.index.union(dementia_counts.index):
        summary_categorical.append({
            'Variable': var,
            'Category': idx,
            'No Dementia (n [%])': f"{no_dementia_counts.get(idx, 0)} ({no_dementia_perc.get(idx, 0.0)}%)",
            'Dementia (n [%])': f"{dementia_counts.get(idx, 0)} ({dementia_perc.get(idx, 0.0)}%)",
            'p-value': format_p_value(p_val)
        })

summary_categorical_df = pd.DataFrame(summary_categorical)

# Display results using standard print statements
print("Continuous Variables Summary:")
print(summary_continuous_df.to_string(index=False))

print("\nCategorical Variables Summary:")
print(summary_categorical_df.to_string(index=False))


# ### Graphs to check data

# In[69]:


import seaborn as sns

# Draw Plot
plt.figure(figsize=(10,10), dpi= 80) #create figure

#Define Plot type, configure Age categories against Depression, define shade, color, labels and transparency
sns.kdeplot(data_stats_imputed.loc[data_stats_imputed['dementia_diagnosis'] == 0, "age_at_baseline"], shade=True, color="g", label="Dementia_diagnosis=0", alpha=.7)
sns.kdeplot(data_stats_imputed.loc[data_stats_imputed['dementia_diagnosis'] == 1, "age_at_baseline"], shade=True, color="deeppink", label="Dementia_diagnosis=1", alpha=.7)

# Title and show plot
plt.title('Density Plot of Age by Dementia Diagnosis', fontsize=22)
plt.legend()
plt.show()
# Save the boxplot as a file in the output directory
plt.savefig(os.path.join(output_dir, 'Density Plot of Age by Dementia Diagnosis.png'))
plt.close()  # Close the plot to free up memory


# In[70]:


import seaborn as sns

# Draw Plot
plt.figure(figsize=(10,10), dpi= 80) #create figure

#Define Plot type, configure Age categories against Depression, define shade, color, labels and transparency
sns.kdeplot(data_stats_imputed.loc[data_stats_imputed['dementia_diagnosis'] == 0, "years_education"], shade=True, color="g", label="Dementia_diagnosis=0", alpha=.7)
sns.kdeplot(data_stats_imputed.loc[data_stats_imputed['dementia_diagnosis'] == 1, "years_education"], shade=True, color="deeppink", label="Dementia_diagnosis=1", alpha=.7)

# Title and show plot
plt.title('Density Plot of Education years by Dementia Diagnosis', fontsize=22)
plt.legend()
plt.show()
# Save the boxplot as a file in the output directory
plt.savefig(os.path.join(output_dir, 'Density Plot of Education years by Dementia Diagnosi.png'))
plt.close()  # Close the plot to free up memory


# In[71]:


import seaborn as sns

# Draw Plot
plt.figure(figsize=(10,10), dpi= 80) #create figure

#Define Plot type, configure Age categories against Depression, define shade, color, labels and transparency
sns.kdeplot(data_stats_imputed.loc[data_stats_imputed['dementia_diagnosis'] == 0, "APOE4_alleles"], shade=True, color="g", label="Dementia_diagnosis=0", alpha=.7)
sns.kdeplot(data_stats_imputed.loc[data_stats_imputed['dementia_diagnosis'] == 1, "APOE4_alleles"], shade=True, color="deeppink", label="Dementia_diagnosis=1", alpha=.7)

# Title and show plot
plt.title('Density Plot of APOE4 alleles by Dementia Diagnosis', fontsize=22)
plt.legend()
plt.show()
# Save the boxplot as a file in the output directory
plt.savefig(os.path.join(output_dir, 'Density Plot of APOE4 alleles by Dementia Diagnosis.png'))
plt.close()  # Close the plot to free up memory


# In[72]:


import seaborn as sns

# Draw Plot
plt.figure(figsize=(10,10), dpi= 80) #create figure

#Define Plot type, configure Age categories against Depression, define shade, color, labels and transparency
sns.kdeplot(data_stats_imputed.loc[data_stats_imputed['dementia_diagnosis'] == 0, "APOE4"], shade=True, color="g", label="Dementia_diagnosis=0", alpha=.7)
sns.kdeplot(data_stats_imputed.loc[data_stats_imputed['dementia_diagnosis'] == 1, "APOE4"], shade=True, color="deeppink", label="Dementia_diagnosis=1", alpha=.7)

# Title and show plot
plt.title('Density Plot of APOE4 by Dementia Diagnosis', fontsize=22)
plt.legend()
plt.show()
# Save the boxplot as a file in the output directory
plt.savefig(os.path.join(output_dir, 'Density Plot of APOE4 by Dementia Diagnosis.png'))
plt.close()  # Close the plot to free up memory


# In[73]:


data_stats_imputed.isnull().sum()


# In[74]:


print(data_stats_imputed.columns.tolist())


# ### Imputing missing data with KNN

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

# Check data types of all columns
data_types = data_stats_imputed.dtypes

# Identify categorical variables
categorical_columns = data_types[data_types == 'object'].index.tolist()
print(f'Categorical columns: {categorical_columns}')

# Convert the 'APOE4_alleles' column to categorical
data_stats_imputed['APOE4_alleles'] = data_stats_imputed['APOE4_alleles'].astype('category')

# Add 'APOE4_alleles' to the list of categorical columns if not already present
if 'APOE4_alleles' not in categorical_columns:
    categorical_columns.append('APOE4_alleles')

# Identify numeric columns
numeric_columns = [col for col in data_stats_imputed if col not in categorical_columns]

# Replace any infinity values (np.inf and -np.inf) with NaN
data_stats_imputed.replace([np.inf, -np.inf], np.nan, inplace=True)

# Convert non-numeric values in numeric columns to NaN
for col in numeric_columns:
    data_stats_imputed[col] = pd.to_numeric(data_stats_imputed[col], errors='coerce')

# Apply KNN imputation for numeric features
knn_imputer = KNNImputer(n_neighbors=5)
data_stats_imputed[numeric_columns] = knn_imputer.fit_transform(data_stats_imputed[numeric_columns])

# Save the imputed dataset to a CSV file
data_stats_imputed.to_csv('data_stats_imputed_knn.csv', index=False)

# Verify the imputation
print("KNN imputation completed and dataset saved to 'data_stats_imputed_knn.csv'.")

