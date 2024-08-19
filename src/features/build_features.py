import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from imblearn.over_sampling import SMOTE


def load_data(file_path):
    """Loads the dataset from the specified file path."""
    return pd.read_csv(file_path)

def save_processed_data(df, output_filepath):
    """
    Save the processed DataFrame to a specified file path.
    
    """
    df.to_csv(output_filepath, index=False)

def encode_categorical_variables(df, categorical_columns):
    """Encodes categorical variables."""
        # Encode ordinal variables
    le = LabelEncoder()
    for column in categorical_columns:
        df[column] = le.fit_transform(df[column])
        
    return df

def encode_music_effects(df):
    """Encode the music_effects column with custom values."""
    # Define the custom mapping
    mapping = {
        'Worsen': -1,
        'No effect': 0,
        'Improve': 1
    }
    
    # Apply the custom mapping to the 'music_effects' column
    df['music_effects'] = df['music_effects'].map(mapping).astype('int')
    
    return df

def create_binary_columns(df):
    """Creates binary columns based on reported levels above 5."""
    conditions = {
        'has_ocd': df['ocd'] > 5,
        'has_depression': df['depression'] > 5,
        'has_insomnia': df['insomnia'] > 5,
        'has_anxiety': df['anxiety'] > 5
    }
    
    for col, condition in conditions.items():
        df[col] = condition.astype(int)
    
    return df

def standard_scale(df, numeric_columns):
    """Standard scales the numerical variables."""
    scaler = StandardScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    return df

def oversample_minority_classes(df, target_column):
    """Oversamples the minority classes."""
    smote = SMOTE(sampling_strategy='not majority', random_state=42)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    df_resampled = pd.concat([X_resampled, y_resampled], axis=1)
    return df_resampled


def main():
    data_filepath = "C:/Users/sirin/Desktop/Github projects/Music-and-Mental-Health/data/interim/music_v_mental_health_cleaned.csv"
    output_filepath = "C:/Users/sirin/Desktop/Github projects/Music-and-Mental-Health/data/processed/music_v_mental_health_processed.csv"

    # Load data
    df = load_data(data_filepath)
    
    # Encode categorical variables
    df = encode_music_effects(df)

    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    df = encode_categorical_variables(df, categorical_columns)
    
    # Create binary columns
    df = create_binary_columns(df)
    
    # Standard scale numerical variables
    numeric_columns = ['age', 'hours_per_day', 'bpm', 'anxiety',
                        'depression', 'insomnia', 'ocd']
    df = standard_scale(df, numeric_columns)
    
    # Oversample minority classes
    target_column = 'music_effects'
    df = oversample_minority_classes(df, target_column)
    
    # Save the processed dataframe to a new file
    df.to_csv('processed_data.csv', index=False)

     # Save Cleaned Data
    save_processed_data(df, output_filepath)
    
    print("Feature engineering complete. Processed data saved to 'processed_data.csv'.")


if __name__ == "__main__":
    main()
