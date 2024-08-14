---
noteId: "d318b4205a8011ef946e63084737c10c"
tags: []

---

# Data Visualization Script Documentation
**Author**: Sirine Maaroufi  
**Date**: August 2024
---
## Overview
This script is designed to perform data visualization on a cleaned dataset to explore the relationship between music preferences, listening habits, and mental health. It generates various plots to help understand patterns and insights related to music consumption and its impact on mental well-being.

## Dependencies
- *matplotlib.pyplot* for plotting graphs.
- *seaborn* for enhanced visualizations.
- *pandas* for data manipulation.
- *os* for file path operations.
- *sklearn.preprocessing.LabelEncoder* for encoding categorical labels.

## Function Definitions

### `load_data(filepath)` 
Loads the dataset from a specified file path.

### `save_plot(filename)`
Saves the current plot to a file.

### `plot_age_distribution(df, output_dir)`
Generates and saves a histogram of age distribution from the dataset.

### `plot_streaming_service(df, output_dir)`
Generates and saves a bar plot of streaming service usage percentages.

### `plot_musicianship_disctribution(df, output_dir)`
Generates and saves a bar plot showing the proportion of responses across two categories that reflect musicianship.

### `plot_music_engagement_disctribution(df, output_dir)`
Generates and saves a bar plot showing the proportion of responses regarding music engagement.

### `plot_favorite_genre(df, output_dir)`
Generates and saves a bar plot of favorite music genres with percentages.

### `plot_while_working(df, output_dir)`
Generates and saves a bar plot showing the percentage of respondents who listen to music while working.

### `plot_hours_per_day(df, output_dir)`
Generates and saves a histogram of hours per day spent listening to music.

### `plot_bpm_distribution(df, output_dir)`
Generates and saves a histogram of preferred BPM (Beats Per Minute) distribution.

### `plot_age_group_genre(df, output_dir)`
Generates and saves a count plot showing the favorite music genre by age group.

### `plot_genre_frequency_counts(df, output_dir)`
Generates and saves a bar plot showing the listening frequency counts for each genre.

### `plot_mental_illnesses(df, output_dir)`
Generates and saves histograms showing the distribution of levels for different mental illnesses.

### `plot_music_effects_pie_chart(df, output_dir)`
Generates and saves a pie chart showing the distribution of music effects on mental health.

### `plot_favorite_genre_and_music_effects(df, output_dir)`
Generates and saves a stacked bar plot showing the relationship between favorite music genre and music effects.


### `plot_bpm_range_and_music_effects(df, output_dir)`
Generates and saves a box plot showing the relationship between BPM and music effects.

### `plot_hours_per_day_and_music_effects(df, output_dir)`
Generates and saves a box plot showing the relationship between hours spent listening to music per day and music effects.

### `plot_instrumentalist_composer_and_music_effects(df, output_dir)`
Generates and saves count plots showing the relationship between being an instrumentalist/composer and music effects.

### `plot_mental_health_and_music_effects(df, output_dir)`
Generates and saves box plots and a combined box plot showing the effect of music on different mental illnesses.

### `create_mental_health_columns(df)`
Creates new boolean columns for each mental health issue based on specific thresholds.

### `visualize_music_preferences_for_mental_illness(df, mental_health_columns, output_dir, music_pref_col)`
Visualizes the preferred genre of music for each mental illness based on boolean columns, ordered from most preferred to least preferred.

### `visualize_percentage_of_mental_illness(df, mental_health_columns, output_dir)`
Visualizes the percentage of individuals with each mental illness based on boolean columns.

### `visualize_hours_vs_music_effects(df, output_dir)`
Visualizes the relationship between hours per day listening to music and music effects.

### `label_encode(df)`
Performs label encoding on specified categorical columns of the DataFrame.


### `plot_correlation_matrix(df, output_dir)`
Generates and saves a heatmap of the correlation matrix for the given DataFrame.


## Main Script Execution

### `main()`
The main function that orchestrates the data loading, visualization generation, and saving of plots.

**Actions:**
- Defines file paths and output directories.
- Loads the data.
- Generates and saves various plots.
- Prints a success message upon completion.

**Note:** Ensure that the necessary directories and file paths are correctly set before running the script.
