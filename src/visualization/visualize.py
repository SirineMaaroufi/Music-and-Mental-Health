import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# --------------------------------------------------------------
# Load data
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

# --------------------------------------------------------------
# Descriptives Statistics
# --------------------------------------------------------------
print(data.describe())
# --------------------------------------------------------------



# --------------------------------------------------------------
# Main function
# --------------------------------------------------------------
if __name__ == "__main__":
    # Define file paths
    input_filepath = 'C:/Users/sirin/Desktop/Music_v_Mental_Health/data/processed/music_v_mental_health_cleaned.csv'

    # Step 1: Load Data
    data = load_data(input_filepath)