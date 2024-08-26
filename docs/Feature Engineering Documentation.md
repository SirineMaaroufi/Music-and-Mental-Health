---
noteId: "971d15c05fc011efa48aa3884b971d03"
tags: []

---

# Feature Engineering Documentation
**Author**: Sirine Maaroufi  
**Date**: August 2024
---

## Overview
This document outlines the steps and techniques used for feature engineering in the preparation of the Music and Mental Health dataset. The goal was to transform raw data into a form suitable for modeling, focusing on encoding categorical variables, creating binary features, scaling numerical features, and handling class imbalance.

## 1. Data Loading
- **Function**: `load_data(file_path)`
- **Description**: The dataset was loaded from a CSV file using `pandas`. This step serves as the initial point for data preparation.

## 2. Encoding the Target Variable (`music_effects`)
- **Function**: `encode_music_effects(df)`
- **Description**: The `music_effects` column, which represents the effect of music on mental health, was encoded with custom values:
  - 'Worsen' → -1
  - 'No effect' → 0
  - 'Improve' → 1
- **Purpose**: To transform the target variable into a numerical format with meaningful ordinal values, aiding in model interpretability and performance.

## 3. Encoding Categorical Variables
- **Function**: `encode_categorical_variables(df, categorical_columns)`
- **Description**: Categorical variables in the dataset were encoded using the [`LabelEncoder`](https://www.geeksforgeeks.org/ml-label-encoding-of-datasets-in-python/). This conversion of categorical data into numerical format allows machine learning models to interpret these features.
- **Columns Encoded**: All categorical columns identified via `df.select_dtypes(include=['object', 'category'])`.

## 4. Creating Binary Columns for Mental Health Conditions
- **Function**: `create_binary_columns(df)`
- **Description**: Binary columns were created to indicate the presence of mental health conditions based on a threshold:
  - Conditions: `ocd`, `depression`, `insomnia`, `anxiety`
  - Binary Representation: If the reported level is greater than 5, it is considered as having the condition (`1`); otherwise, not (`0`).
- **Purpose**: To simplify the interpretation of mental health conditions by creating binary indicators.

## 5. Standard Scaling of Numerical Variables
- **Function**: `standard_scale(df, numeric_columns)`
- **Description**: Numerical variables were standardized using `StandardScaler`. This ensures that each feature has a mean of 0 and a standard deviation of 1.
- **Columns Scaled**: 
  - `age`
  - `hours_per_day`
  - `bpm`
  - `anxiety`
  - `depression`
  - `insomnia`
  - `ocd`
- **Purpose**: Scaling was performed to normalize the range of numerical features, which can improve the performance of many machine learning algorithms.

## 6. Handling Class Imbalance through Oversampling
- **Function**: `oversample_minority_classes(df, target_column)`
- **Description**: To address class imbalance in the `music_effects` target variable, [SMOTE (Synthetic Minority Over-sampling Technique)](https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/) was applied. This technique generates synthetic samples for the minority classes to ensure a more balanced class distribution.
- **Resulting Dataset**: 
  - **Rows**: 1626
  - **Columns**: 35
  - **Music Effects Distribution**: Each category ('Worsen', 'No effect', 'Improve') has **542 entries.**
- **Purpose**: Oversampling helps in mitigating bias towards the majority class during model training, leading to better generalization.

## 7. Correlation Analysis
- **Function**: `correlation_analysis(df, target_column, threshold=0.5, save_path='correlation_matrix.png')`
- **Description**: To identify relationships between features and the target variable, a correlation analysis was conducted. The analysis computed the correlation matrix and visualized it using a heatmap, which was saved as an image for reference.
![Correlation matrix](/reports/Figures/correlation_matrix_post_FE.png) 
### 7.1. Highly Correlated Features
During the analysis, certain pairs of features were found to have high correlation values (greater than 0.5 in absolute value). These pairs might lead to redundancy in the data, which can affect some machine learning models. Below are the identified pairs of highly correlated features:
  - `freq_rnb` and `freq_hiphop`
  - `freq_rap` and `freq_hiphop`
  - `freq_rock` and `freq_metal`
  - `has_ocd` and `ocd`
  - `has_depression` and `depression`
  - `has_insomnia` and `insomnia`
  - `has_anxiety` and `anxiety `

These pairs indicate that listeners of R&B often listen to hip-hop, and similarly for other genres. Additionally, binary indicators of mental health conditions are highly correlated with their corresponding severity scores.
### 7.2. Correlation with target value
The correlation analysis also highlighted which features have the strongest relationships with the target variable music_effects. The following features exhibited the highest correlations:\
* Positive Correlations:
  - `instrumentalist`: 0.34
  - `while_working`: 0.34
  - `freq_jazz`: 0.31
  - `exploratory`: 0.30
  - `freq_rnb`: 0.29

* Negative Correlations:
  - `depression`: -0.32
  - `favorite_genre`: -0.33

These correlations suggest that individuals who play musical instruments or listen to jazz music tend to report a positive effect of music on mental health. Conversely, those with higher depression scores or specific favorite genres might report less positive effects.

## 8. Saving the Processed Data
- **Function**: `save_processed_data(df, output_filepath)` 
- **Description**: The processed dataset was saved to a specified file path for subsequent modeling steps.

## Conclusion
The feature engineering process effectively transformed the raw dataset into a structured, numerical format suitable for machine learning. Key steps included encoding categorical variables, creating binary features, standard scaling, and addressing class imbalance. The resulting dataset, consisting of 1626 rows and 35 columns, with balanced classes in the `music_effects` variable, is now ready for model development and evaluation.

The correlation analysis provided valuable insights into feature relationships and their impact on the target variable, guiding the selection of features for subsequent modeling. These steps ensure that the model is trained on the most informative and non-redundant data, improving its accuracy and robustness.


