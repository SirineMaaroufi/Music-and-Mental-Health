"""
This script performs data preparation tasks on the Music and Mental Health dataset.
It includes data loading, cleaning, handling missing values and outliers, renaming columns, 
adjusting data types, and saving the cleaned data for further analysis.

Author: Sirine Maaroufi
Date: August 2024
"""

import pandas as pd
from sklearn.impute import SimpleImputer

# --------------------------------------------------------------
# Function Definitions
# --------------------------------------------------------------

def load_data(filepath):
    """
    Load the dataset from a specified file path.
    
    Parameters:
    filepath (str): The path to the CSV file.
    
    Returns:
    pd.DataFrame: Loaded dataset as a DataFrame.
    """
    return pd.read_csv(filepath)


def drop_unnecessary_columns(df, columns):
    """
    Drop specified columns from the DataFrame.
    
    Parameters:
    df (pd.DataFrame): The original DataFrame.
    columns (list): List of columns to be dropped.
    
    Returns:
    pd.DataFrame: DataFrame after dropping unnecessary columns.
    """
    return df.drop(columns=columns)


def rename_columns(df, new_column_names):
    """
    Rename the columns of the DataFrame based on a provided mapping.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to rename columns.
    new_column_names (dict): Dictionary mapping old column names to new names.
    
    Returns:
    pd.DataFrame: DataFrame with renamed columns.
    """
    return df.rename(columns=new_column_names)


def adjust_data_types(df, categorical_columns, integer_columns):
    """
    Adjust data types for the DataFrame's columns.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to adjust data types.
    categorical_columns (list): List of columns to convert to 'category' type.
    integer_columns (list): List of columns to convert to 'int' type.
    
    Returns:
    pd.DataFrame: DataFrame with adjusted data types.
    """
    df[categorical_columns] = df[categorical_columns].astype('category')
    df[integer_columns] = df[integer_columns].astype('int')
    return df


def save_cleaned_data(df, filepath):
    """
    Save the cleaned DataFrame to a specified file path.
    
    Parameters:
    df (pd.DataFrame): The cleaned DataFrame to be saved.
    filepath (str): The path to save the cleaned CSV file.
    """
    df.to_csv(filepath, index=False)


def impute_missing_categorical(df):
    """
    Detect categorical columns with missing values and impute them with the most frequent category.

    Parameters:
    df (pd.DataFrame): The input DataFrame with potential missing values in categorical columns.

    Returns:
    pd.DataFrame: DataFrame with missing values in categorical columns imputed.
    """
    categorical_columns = df.select_dtypes(include=['object']).columns
    imputer = SimpleImputer(strategy='most_frequent')
    
    for col in categorical_columns:
        if df[col].isna().sum() > 0:
            df[col] = imputer.fit_transform(df[[col]]).ravel()
    
    return df


def impute_bpm(row, bpm_mapping):
    """
    Impute BPM values based on the favorite genre using a predefined mapping.

    Parameters:
    row (pd.Series): A row of the DataFrame.
    bpm_mapping (dict): Dictionary mapping genres to average BPM values.

    Returns:
    float: Imputed BPM value.
    """
    
    if pd.isna(row['bpm']):
        return bpm_mapping.get(row['favorite_genre'], row['bpm'])
    else: 
        return row['bpm']


def impute_missing_age(df):
    """
    Impute the age column with the mean.

    Parameters:
    df (pd.DataFrame): The input DataFrame with potential missing values in the age column.

    Returns:
    pd.DataFrame: DataFrame with missing values in the age column imputed.
    """
    imputer = SimpleImputer(strategy='mean')
    df['age'] = imputer.fit_transform(df[['age']])
    return df


def detect_outliers_iqr(df, columns):
    """
    Detect outliers in the specified columns of a DataFrame using the Interquartile Range (IQR) method.
    Returns a summary of the number of outliers and their values.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to check for outliers.
    columns (list): List of columns to check for outliers.
    
    Returns:
    dict: A dictionary where keys are column names and values are tuples containing:
          - A DataFrame with outliers flagged.
          - A DataFrame showing the outliers' values and counts.
    """
    # Initialize result dictionary
    result = {}
    
    for column in columns:
        # Calculate Q1 (25th percentile) and Q3 (75th percentile)
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        
        # Calculate the Interquartile Range (IQR)
        IQR = Q3 - Q1
        
        # Define outlier bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Flag outliers
        df_flag = df.copy()
        df_flag[f'{column}_outlier'] = df_flag[column].apply(lambda x: x < lower_bound or x > upper_bound)
        
        # Get the outliers
        outliers_df = df_flag[df_flag[f'{column}_outlier']][[column, f'{column}_outlier']]
        
        # Count of outliers
        num_outliers = outliers_df.shape[0]
        
        # Store results in the dictionary
        result[column] = (df_flag, outliers_df, num_outliers)
    
    return result


def outliers_display(results):
    for column, (df_with_flags, outliers_df, count) in results.items():
        print(f"Column: {column}")
        print(f"Number of outliers: {count}")
        print("Outliers:")
        print(outliers_df)
        
        
def handle_outliers_bpm(row, bpm_mapping):
    """
    Impute BPM values based on the favorite genre using a predefined mapping.

    Parameters:
    row (pd.Series): A row of the DataFrame.
    bpm_mapping (dict): Dictionary mapping genres to average BPM values.

    Returns:
    float: Imputed BPM value.
    """
    if row['bpm'] < 10:
        return bpm_mapping.get(row['favorite_genre'], row['bpm'])
    if row['bpm'] > 240:
        return bpm_mapping.get(row['favorite_genre'], row['bpm'])
    return row['bpm']

# --------------------------------------------------------------
# Main Script Execution
# --------------------------------------------------------------

if __name__ == "__main__":
    # Define file paths
    input_filepath = 'C:/Users/sirin/Desktop/Music_v_Mental_Health/data/raw/mxmh_survey_results.csv'
    output_filepath = 'C:/Users/sirin/Desktop/Music_v_Mental_Health/data/processed/music_v_mental_health_cleaned.csv'

    # Load Data
    data = load_data(input_filepath)
    
    # Drop Unnecessary Columns
    columns_to_drop = ['Permissions', 'Timestamp']
    data_cleaned = drop_unnecessary_columns(data, columns_to_drop)
    
    # Rename Columns
    new_column_names = {
        'Age': 'age',
        'Primary streaming service': 'streaming_service',
        'Hours per day': 'hours_per_day',
        'While working': 'while_working',
        'Instrumentalist': 'instrumentalist',
        'Composer': 'composer',
        'Fav genre': 'favorite_genre',
        'Exploratory': 'exploratory',
        'Foreign languages': 'foreign_languages',
        'BPM': 'bpm',
        'Frequency [Classical]': 'freq_classical',
        'Frequency [Country]': 'freq_country',
        'Frequency [EDM]': 'freq_edm',
        'Frequency [Folk]': 'freq_folk',
        'Frequency [Gospel]': 'freq_gospel',
        'Frequency [Hip hop]': 'freq_hiphop',
        'Frequency [Jazz]': 'freq_jazz',
        'Frequency [K pop]': 'freq_kpop',
        'Frequency [Latin]': 'freq_latin',
        'Frequency [Lofi]': 'freq_lofi',
        'Frequency [Metal]': 'freq_metal',
        'Frequency [Pop]': 'freq_pop',
        'Frequency [R&B]': 'freq_rnb',
        'Frequency [Rap]': 'freq_rap',
        'Frequency [Rock]': 'freq_rock',
        'Frequency [Video game music]': 'freq_video_game',
        'Anxiety': 'anxiety',
        'Depression': 'depression',
        'Insomnia': 'insomnia',
        'OCD': 'ocd',
        'Music effects': 'music_effects'
    }
    data_cleaned = rename_columns(data_cleaned, new_column_names)
    
    # Handle Missing Values
    ## Drop Missing values in 'music_effects'
    data_cleaned = data_cleaned.dropna(subset=['music_effects']).reset_index(drop=True)
    
    # Define BPM mapping
    bpm_mapping = {
        'Folk': (80 + 120) / 2,
        'Country': (100 + 140) / 2,
        'Metal': (100 + 160) / 2,
        'Hip hop': (80 + 115) / 2,
        'Rap': (80 + 115) / 2,
        'Pop': (100 + 130) / 2,
        'R&B': (60 + 100) / 2,
        'Rock': (110 + 140) / 2,
        'Lofi': (60 + 90) / 2,
        'Latin': (90 + 140) / 2,
        'K pop': (100 + 130) / 2,
        'Jazz': (100 + 150) / 2,
        'Gospel': (80 + 120) / 2,
        'EDM': (120 + 150) / 2,
        'Classical': (60 + 120) / 2,
        'Video game music': (80 + 160) / 2
    }
    
    ## Impute BPM
    data_cleaned['bpm'] = data_cleaned.apply(impute_bpm, axis=1, bpm_mapping = bpm_mapping)
    
    ## Impute Categorical Values
    data_cleaned = impute_missing_categorical(data_cleaned)
    
    ## Impute Age with Mean
    data_cleaned = impute_missing_age(data_cleaned)
    
    # Adjust Data Types
    categorical_columns = [col for col in data_cleaned.columns if data_cleaned[col].dtype == 'object']
    integer_columns = ['age', 'anxiety', 'depression', 'insomnia', 'ocd']

    data_cleaned = adjust_data_types(data_cleaned, categorical_columns, integer_columns)
    
    # Detect outliers
    results = detect_outliers_iqr(data_cleaned, ['age', 'bpm'])
    #outliers_display(results)    

    
    # Impute Outliers in BPM 
    data_cleaned['bpm'] = data_cleaned.apply(handle_outliers_bpm, axis=1, bpm_mapping = bpm_mapping)
    
    # Save Cleaned Data
    save_cleaned_data(data_cleaned, output_filepath)
    
    print(f"Data cleaning complete. Cleaned data saved to {output_filepath}")
