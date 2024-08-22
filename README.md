# Code used in the dissertation
### Background
This repo is the companion of the MRes dissertation with the title: **Individualised risk prediction for dementia, deriving actionable information from multimodal data**. This research aims to identify cost-effective, non-invasive dementia risk signatures, such as blood-based biomarkers for dementia risk prediction. To the best of the author’s knowledge, this is the first study combining advances in machine learning and integrating proteomics, metabolomics, genetic markers, and baseline clinical variables for individualised dementia risk prediction in the preclinical stage. Results from this work will determine whether blood-based biomarkers and machine learning can predict future dementia risk at an individual level. 

All the code used for this MRes project is available in this repo. Below more information on how it is organised.

### Code in this repo
- **Data pre-processing and descriptive statistics folder** This reflects work done in sections 5.1 and part of 5.2.
  
    - Name of folder: Data_pre_process_and_statsanalysis
      * File name: Data_pre_process_and_statsanalysis.ipynb, also available as a .py script.  This python script performs:
      - Merge of the proteomics and metabolomics datasets
      - Data cleaning
      - Missing data checks and removal of data based on criteria
      - Check for outliers
      - MCAR/MAR assumptions test
      - Summary statistics (section 5.4)
      - Imputation of missing data with KNN
      * GFAP_NEFL_normalised: Script for the two linear regressions done to account for the effect of age in GFAP and NEFL is included in the machine learning as classification folder
      

- **Descriptive statistics**
    - Script for statistical analysis done (section 5.4). This is included in 
 
- **Machine learning as a classification**
    - First experiment set (section 5.4.1.3.1): Name of files: [include name of file]
    - Second experiment set (section 5.4.1.3.2): Name of files: [include name of file]
    - Third experiment set (section 5.4.1.3.3): Name of files: [include name of file]
    - Fourth experiment set (section 5.4.1.3.4): Name of files: [include name of file]
 
- **Survival machine learning**
    -  Code as per described in section 5.4.2. Name of files: [include name of file]
      
 - **Summary of results**  
      Python code for the summary of the results for the first and second round of experiments (section 6.2.1 and 6.2.2).  
    - Name of folder: Summary_results  
    - Name of files: Summary_results.ipynb (there is also a .py version of the same script)
      
      
 
      
