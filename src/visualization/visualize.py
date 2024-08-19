"""
The script is designed to perform data visualization on a cleaned dataset that explores 
the relationship between music preferences, listening habits, and mental health.

Author: Sirine Maaroufi
Date: August 2024
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder


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


def save_plot(filename):
    """
    Save the current plot to a file.

    Parameters:
    filename (str): The path where the plot will be saved.
    """
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


def plot_age_distribution(df, output_dir):
    """
    Generate and save a histogram of age distribution from the dataset.

    Parameters:
    df (pd.DataFrame): The dataset containing age data.
    output_dir (str): The directory where the plot will be saved.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x="age", bins=20, kde=True, color="skyblue")
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    save_plot(os.path.join(output_dir, "age_distribution.png"))


def plot_streaming_service(df, output_dir):
    """
    Generate and save a bar plot of streaming service usage percentages.

    Parameters:
    df (pd.DataFrame): The dataset containing streaming service data.
    output_dir (str): The directory where the plot will be saved.
    """
    # Calculate the counts and percentages of each streaming service
    streaming_counts = df["streaming_service"].value_counts().reset_index()
    streaming_counts.columns = ["Streaming Service", "Count"]
    streaming_counts["Percentage"] = (
        streaming_counts["Count"] / streaming_counts["Count"].sum()
    ) * 100
    # Plot the bar chart
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=streaming_counts, y="Streaming Service", x="Percentage", palette="pastel"
    )
    plt.title("Streaming Service Usage with Percentages")
    plt.xlabel("Streaming Service")
    plt.ylabel("Percentage")
    plt.xticks(rotation=45)
    save_plot(os.path.join(output_dir, "streaming_service_usage.png"))


def plot_musicianship_disctribution(df, output_dir):
    """
    Generate and save a bar plot showing the proportion of responses across two categories that reflect musicicanship.

    Parameters:
    df (pd.DataFrame): The dataset containing categorical response data.
    output_dir (str): The directory where the plot will be saved.
    """
    # Melt the DataFrame to long format
    df_melted = df.melt(
        value_vars=["instrumentalist", "composer"],
        var_name="Category",
        value_name="Response",
    )
    # Calculate counts and percentages
    category_counts = (
        df_melted.groupby(["Category", "Response"]).size().reset_index(name="Count")
    )
    category_totals = df_melted.groupby("Category").size().reset_index(name="Total")
    category_counts = category_counts.merge(category_totals, on="Category")
    category_counts["Percentage"] = (
        category_counts["Count"] / category_counts["Total"]
    ) * 100
    # Plot the bar chart with categories and responses
    plt.figure(figsize=(12, 7))
    sns.barplot(
        data=category_counts,
        x="Category",
        y="Percentage",
        hue="Response",
        palette="pastel",
    )
    plt.title(
        "Percentage of Musicians by Role: Composer and Instrumentalist"
    )
    plt.xlabel("Category")
    plt.ylabel("Percentage")
    plt.legend(title="Response")
    save_plot(os.path.join(output_dir, "musicianship_distribution.png"))

def plot_music_engagement_disctribution(df, output_dir):
    """
    Generates and saves a bar plot showing the proportion of responses regarding music engagement.

    Parameters:
    df (pd.DataFrame): The dataset containing categorical response data.
    output_dir (str): The directory where the plot will be saved.
    """
    # Melt the DataFrame to long format
    df_melted = df.melt(
        value_vars=["exploratory", "foreign_languages"],
        var_name="Category",
        value_name="Response",
    )
    # Calculate counts and percentages
    category_counts = (
        df_melted.groupby(["Category", "Response"]).size().reset_index(name="Count")
    )
    category_totals = df_melted.groupby("Category").size().reset_index(name="Total")
    category_counts = category_counts.merge(category_totals, on="Category")
    category_counts["Percentage"] = (
        category_counts["Count"] / category_counts["Total"]
    ) * 100
    # Plot the bar chart with categories and responses
    plt.figure(figsize=(12, 7))
    sns.barplot(
        data=category_counts,
        x="Category",
        y="Percentage",
        hue="Response",
        palette="pastel",
    )
    plt.title(
        "Percentage of Music Engagement by Exploration and Foreign Language Listening"
    )
    plt.xlabel("Category")
    plt.ylabel("Percentage")
    plt.legend(title="Response")
    save_plot(os.path.join(output_dir, "music_engagement_distribution.png"))
    
def plot_favorite_genre(df, output_dir):
    """
    Generate and save a bar plot of favorite music genres with percentages.

    Parameters:
    df (pd.DataFrame): The dataset containing favorite genre data.
    output_dir (str): The directory where the plot will be saved.
    """
    # Calculate the counts and percentages of each favorite genre

    genre_counts = df["favorite_genre"].value_counts().reset_index()
    genre_counts.columns = ["Favorite Music Genre", "Count"]
    genre_counts["Percentage"] = (
        genre_counts["Count"] / genre_counts["Count"].sum()
    ) * 100
    # Plot the bar chart
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=genre_counts, y="Favorite Music Genre", x="Percentage", palette="pastel"
    )
    plt.title("Favorite Music Genre with Percentages")
    plt.xlabel("Favorite Music Genre")
    plt.ylabel("Percentage")
    plt.xticks(rotation=45)
    save_plot(os.path.join(output_dir, "favorite_genre.png"))


def plot_while_working(df, output_dir):
    """
    Generate and save a bar plot showing the percentage of respondents who listen to music while working.

    Parameters:
    df (pd.DataFrame): The dataset containing data on listening while working.
    output_dir (str): The directory where the plot will be saved.
    """
    # Calculate the counts and percentages of listening to music while working

    while_working_counts = df["while_working"].value_counts().reset_index()
    while_working_counts.columns = ["Listen While Working (Yes/No)", "Count"]
    while_working_counts["Percentage"] = (
        while_working_counts["Count"] / while_working_counts["Count"].sum()
    ) * 100
    # Plot the bar chart
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=while_working_counts,
        x="Listen While Working (Yes/No)",
        y="Percentage",
        palette="pastel",
    )
    plt.title("Listening to Music While Working with Percentages")
    plt.xlabel("Listen While Working (Yes/No)")
    plt.ylabel("Percentage")
    save_plot(os.path.join(output_dir, "while_working.png"))


def plot_hours_per_day(df, output_dir):
    """
    Generate and save a histogram of hours per day spent listening to music.

    Parameters:
    df (pd.DataFrame): The dataset containing hours per day data.
    output_dir (str): The directory where the plot will be saved.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x="hours_per_day", bins=20, kde=True, color="lightcoral")
    plt.title("Hours per Day Listening to Music")
    plt.xlabel("Hours per Day")
    plt.ylabel("Frequency")
    save_plot(os.path.join(output_dir, "hours_per_day.png"))


def plot_bpm_distribution(df, output_dir):
    """
    Generate and save a histogram of preferred BPM (Beats Per Minute) distribution.

    Parameters:
    df (pd.DataFrame): The dataset containing BPM data.
    output_dir (str): The directory where the plot will be saved.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x="bpm", bins=20, kde=True, color="orchid")
    plt.title("Preferred BPM Distribution")
    plt.xlabel("BPM")
    plt.ylabel("Frequency")
    save_plot(os.path.join(output_dir, "bpm_distribution.png"))


def plot_age_group_genre(df, output_dir):
    """
    Generate and save a count plot showing the favorite genre by age group.

    Parameters:
    df (pd.DataFrame): The dataset containing age group and genre data.
    output_dir (str): The directory where the plot will be saved.
    """
    # Bin ages into groups

    age_bins = pd.cut(
        df["age"],
        bins=[10, 20, 30, 40, 50, 60, 70],
        labels=["10-20", "21-30", "31-40", "41-50", "51-60", "61-70"],
        right=False,
    )
    df["age_bins"] = age_bins
    genre_counts = (
        df.groupby(["age_bins", "favorite_genre"], observed=False)
        .size()
        .reset_index(name="Count")
    )
    # Calculate counts of favorite genres by age group

    ordered_genres = pd.DataFrame()
    for age_group in genre_counts["age_bins"].unique():
        subset = genre_counts[genre_counts["age_bins"] == age_group]
        ordered = subset.sort_values(by="Count", ascending=False)[
            "favorite_genre"
        ].tolist()
        temp_df = pd.DataFrame({"age_bins": age_group, "favorite_genre": ordered})
        ordered_genres = pd.concat([ordered_genres, temp_df], axis=0)

    ordered_genres = ordered_genres.reset_index(drop=True)
    plot_data = pd.merge(df, ordered_genres, on=["age_bins", "favorite_genre"])
    plot_data["favorite_genre"] = pd.Categorical(
        plot_data["favorite_genre"],
        categories=ordered_genres["favorite_genre"].unique(),
        ordered=True,
    )

    plt.figure(figsize=(14, 8))
    sns.countplot(
        data=plot_data,
        x="age_bins",
        hue="favorite_genre",
        palette="tab20",
        order=["10-20", "21-30", "31-40", "41-50", "51-60", "61-70"],
    )
    plt.title("Favorite Genre by Age Group")
    plt.xlabel("Age Group")
    plt.ylabel("Count")
    plt.legend(title="Favorite Genre", bbox_to_anchor=(1.05, 1), loc="upper left")
    save_plot(os.path.join(output_dir, "age_group_genre.png"))


def plot_genre_frequency_counts(df, output_dir):
    # Filter columns that contain 'freq'
    freq_columns = [col for col in df.columns if "freq" in col]
    
     # Melt the DataFrame to long format for easier plotting
    df_long = df.melt(value_vars=freq_columns, var_name='Genre', value_name='Frequency')
    
    # Aggregate counts for each frequency category
    genre_freq_counts = df_long.groupby(['Genre', 'Frequency']).size().reset_index(name='Count')
    
    # Define the order for frequency categories
    frequency_order = ['Very frequently', 'Sometimes', 'Rarely', 'Never']
    
    # Ensure the frequency category is ordered
    genre_freq_counts['Frequency'] = pd.Categorical(genre_freq_counts['Frequency'], categories=frequency_order, ordered=True)
    
    # Clean up the Genre column by removing 'freq_' prefix
    genre_freq_counts['Genre'] = genre_freq_counts['Genre'].str.replace('freq_', '', regex=False)
    
    # Plot
    plt.figure(figsize=(14, 10))
    sns.barplot(
        data=genre_freq_counts,
        x='Genre',
        y='Count',
        hue='Frequency',
        palette='coolwarm',
        hue_order=frequency_order
    )
    plt.title('Listening Frequency Counts for Each Genre')
    plt.xlabel('Music Genre')
    plt.ylabel('Count')
    plt.legend(title='Frequency', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, "genre_frequency.png"))

def plot_mental_illnesses(df, output_dir):
    """
    Generate and save histograms showing the distribution of levels for different mental illnesses.

    Parameters:
    df (pd.DataFrame): The dataset containing mental illness level data.
    output_dir (str): The directory where the plots will be saved.
    """
    # List of mental illnesses to plot

    mental_illnesses = ["anxiety", "depression", "ocd", "insomnia"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    # Create subplots for each mental illness

    plt.figure(figsize=(12, 8))
    for i, illness in enumerate(mental_illnesses, 1):
        plt.subplot(2, 2, i)
        sns.histplot(
            df[illness], bins=11, kde=False, color="skyblue", edgecolor="black"
        )
        plt.title(f"Distribution of {illness.capitalize()} Levels")
        plt.xlim(0, 10)
        plt.xlabel("Level")
        plt.ylabel("Count")
    plt.tight_layout()
    save_plot(os.path.join(output_dir, "mental_illnesses_distribution.png"))
    # Create a combined histogram for all mental illnesses

    plt.figure(figsize=(10, 6))
    for illness, color in zip(mental_illnesses, colors):
        sns.histplot(
            df[illness], bins=11, kde=False, color=color, label=illness.capitalize()
        )
    plt.title("Distribution of Mental Illness Levels")
    plt.xlabel("Level")
    plt.ylabel("Count")
    plt.legend(title="Mental Illness")
    save_plot(os.path.join(output_dir, "mental_illnesses_combined.png"))


def plot_music_effects_pie_chart(df, output_dir):
    """
    Generate and save a pie chart showing the distribution of music effects on mental health.

    Parameters:
    df (pd.DataFrame): The dataset containing music effects data.
    output_dir (str): The directory where the plot will be saved.
    """
    # Define custom colors
    global colors_custom
    colors_custom = {"Improve": "#90EE90", "No effect": "#ADD8E6", "Worsen": "#FFCCCB"}

    # Get the value counts and categories
    value_counts = df["music_effects"].value_counts()
    categories = value_counts.index

    # Map the categories to the custom colors
    colors_map = {
        category: colors_custom.get(category, "#ADD8E6") for category in categories
    }
    pie_colors = [colors_map[category] for category in categories]

    # Plot the pie chart
    plt.figure(figsize=(8, 8))
    df["music_effects"].value_counts().plot.pie(
        autopct="%1.1f%%", colors=pie_colors, startangle=90
    )
    plt.title("Distribution of Music Effects on Mental Health")
    plt.ylabel("")
    save_plot(os.path.join(output_dir, "music_effects_pie.png"))


def plot_favorite_genre_and_music_effects(df, output_dir):
    """
    Generate and save a stacked bar plot showing the relationship between favorite music genre and music effects.

    Parameters:
    df (pd.DataFrame): The dataset containing favorite genre and music effects data.
    output_dir (str): The directory where the plot will be saved.
    """
    # Calculate the counts of 'Improve' for each genre
    improve_counts = (
        df[df["music_effects"] == "Improve"].groupby("favorite_genre").size()
    )

    # Sort genres based on 'Improve' counts
    sorted_genres = improve_counts.sort_values(ascending=False).index

    # Create a new column 'favorite_genre_ordered' for the sorted order
    df["favorite_genre_ordered"] = pd.Categorical(
        df["favorite_genre"], categories=sorted_genres, ordered=True
    )

    # Define the order for music_effects
    music_effects_order = ["Improve", "No effect", "Worsen"]
    df["music_effects"] = pd.Categorical(
        df["music_effects"], categories=music_effects_order, ordered=True
    )

    # Plot the stacked bar plot
    plt.figure(figsize=(12, 8))
    sns.countplot(
        data=df, y="favorite_genre_ordered", hue="music_effects", palette=colors_custom
    )
    plt.title("Music Effects by Favorite Genre")
    plt.xlabel("Count")
    plt.ylabel("Favorite Genre")
    plt.legend(title="Music Effects", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xticks(rotation=45)
    save_plot(os.path.join(output_dir, "fav_genre_music_effect.png"))


def plot_bpm_range_and_music_effects(df, output_dir):
    """
    Generate and save a box plot showing the relationship between BPM and music effects.

    Parameters:
    df (pd.DataFrame): The dataset containing BPM and music effects data.
    output_dir (str): The directory where the plot will be saved.
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="music_effects", y="bpm", palette=colors_custom)
    plt.title("Relationship Between BPM and Music Effects")
    plt.xlabel("Music Effects")
    plt.ylabel("BPM")
    save_plot(os.path.join(output_dir, "bpm_music_effects.png"))


def plot_hours_per_day_and_music_effects(df, output_dir):
    """
    Generate and save a box plot showing the relationship between hours per day of listening to music and music effects.

    Parameters:
    df (pd.DataFrame): The dataset containing hours per day and music effects data.
    output_dir (str): The directory where the plot will be saved.
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="music_effects", y="hours_per_day", palette=colors_custom)
    plt.title(
        "Relationship Between Number of Listening Hours per Day and Music Effects"
    )
    plt.xlabel("Music Effects")
    plt.ylabel("Hours per Day")
    save_plot(os.path.join(output_dir, "hr_per_day_music_effects.png"))


def plot_instrumentalist_composer_and_music_effects(df, output_dir):
    """
    Generate and save count plots showing the relationship between being an instrumentalist/composer and music effects.

    Parameters:
    df (pd.DataFrame): The dataset containing data on being an instrumentalist/composer and music effects.
    output_dir (str): The directory where the plots will be saved.
    """
    for column in ["instrumentalist", "composer"]:
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x=column, hue="music_effects", palette=colors_custom)
        plt.title(
            f"Relationship Between being a musician ({column.capitalize()}) and Music Effects"
        )
        plt.xlabel(column.capitalize())
        plt.ylabel("Count")
        plt.legend(title="Music Effects", bbox_to_anchor=(1.05, 1), loc="upper left")
        save_plot(os.path.join(output_dir, f"{column}_music_effects.png"))


def plot_mental_health_and_music_effects(df, output_dir):
    """
    Generate and save box plots and a combined box plot showing the effect of music on different mental illnesses.

    Parameters:
    df (pd.DataFrame): The dataset containing mental illness levels and music effects data.
    output_dir (str): The directory where the plots will be saved.
    """
    mental_health_issues = ["anxiety", "depression", "ocd", "insomnia"]
    # Create box plots for each mental illness

    for issue in mental_health_issues:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x="music_effects", y=issue, palette=colors_custom)
        plt.title(f"Music Effects on {issue.capitalize()} Levels")
        plt.xlabel("Music Effects")
        plt.ylabel(f"{issue.capitalize()} Level")
        save_plot(os.path.join(output_dir, f"{issue}_music_effects.png"))

    # Create box plots for each mental illness combined in one figure
    melted_df = df.melt(
        id_vars=["music_effects"],
        value_vars=mental_health_issues,
        var_name="Mental Health Issue",
        value_name="Level",
    )
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=melted_df,
        x="music_effects",
        y="Level",
        hue="Mental Health Issue",
        palette="pastel",
    )
    plt.title("Music effect on Mental Illnesses")
    plt.xlabel("Music Effects")
    plt.ylabel("Level")
    plt.legend(title="Mental Health Issue", bbox_to_anchor=(1.05, 1), loc="upper left")
    save_plot(os.path.join(output_dir, "mental_illness_music_effects_combined.png"))

def create_mental_health_columns(df):
    """
    Creates new boolean columns for each mental health issue based on specific thresholds.
    
    Parameters:
    - df: DataFrame containing the dataset with columns 'ocd', 'insomnia', 'depression', 'anxiety'.
    
    Returns:
    - DataFrame with new boolean columns.
    """
    # Define thresholds
    ocd_threshold = 5
    insomnia_threshold = 5
    depression_threshold = 5
    anxiety_threshold = 5

    # Create boolean columns based on thresholds
    df['has_ocd'] = df['ocd'] > ocd_threshold
    df['has_insomnia'] = df['insomnia'] > insomnia_threshold
    df['has_depression'] = df['depression'] > depression_threshold
    df['has_anxiety'] = df['anxiety'] > anxiety_threshold

    return df

def visualize_music_preferences_for_mental_illness(df, mental_health_columns, output_dir,
                                                   music_pref_col):
    """
    Visualizes the preferred genre of music for each mental illness based on boolean columns,
    ordered from most preferred to least preferred.
    
    Parameters:
    - df: DataFrame containing the dataset with new boolean columns.
    - mental_health_columns: List of boolean columns representing different mental health issues.
    - music_pref_col: The column name representing the music preference (default is 'favorite_genre').
    - palette: The color palette to use for the plot.
    """
    # Melt the DataFrame to long format for seaborn
    df_melted = df.melt(id_vars=[music_pref_col], value_vars=mental_health_columns,
                        var_name='Mental Health Issue', value_name='Has Issue')
    
    # Filter out rows where 'Has Issue' is False
    df_melted = df_melted[df_melted['Has Issue']]
    
    # Calculate the counts of each genre for each mental health issue
    genre_counts = df_melted.groupby(['Mental Health Issue', music_pref_col]).size().reset_index(name='Count')
    
    # For each mental health issue, sort genres by count in descending order
    sorted_genre_counts = genre_counts.sort_values(by=['Mental Health Issue', 'Count'], ascending=[True, False])
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot the count of favorite genres for each mental health issue
    sns.barplot(data=sorted_genre_counts, y=music_pref_col, x='Count', hue='Mental Health Issue', palette='pastel')
    
    plt.title('Preferred Music Genre for Each Mental Illness (Ordered by Preference)')
    plt.xlabel('Count')
    plt.ylabel('Favorite Genre')
    plt.legend(title='Mental Health Issue', bbox_to_anchor=(1.05, 1), loc='upper left')
    save_plot(os.path.join(output_dir, "music_prefrences_mental_illness.png"))

    
def visualize_percentage_of_mental_illness(df, mental_health_columns, output_dir):
    """
    Visualizes the percentage of individuals with each mental illness based on boolean columns.
    
    Parameters:
    - df: DataFrame containing the dataset with new boolean columns.
    - mental_health_columns: List of boolean columns representing different mental health issues.
    - palette: The color palette to use for the plot.
    """
    # Calculate the percentage of individuals with each mental illness
    total_count = len(df)
    illness_percentages = {col: (df[col].sum() / total_count) * 100 for col in mental_health_columns}
    
    # Convert to DataFrame for plotting
    percentage_df = pd.DataFrame(list(illness_percentages.items()), columns=['Mental Health Issue', 'Percentage'])
    
    # Plot the data
    plt.figure(figsize=(10, 6))
    sns.barplot(data=percentage_df, x='Mental Health Issue', y='Percentage', palette='pastel')
    
    plt.title('Percentage of Individuals with Each Mental Illness')
    plt.xlabel('Mental Health Issue')
    plt.ylabel('Percentage (%)')
    plt.ylim(0, 100)  # Ensure the y-axis goes from 0 to 100%
    save_plot(os.path.join(output_dir, "percentage_mental_illness.png"))
    
def visualize_hours_vs_music_effects(df,output_dir):
    """
    Visualizes the relationship between hours per day listening to music and music effects.
    
    Parameters:
    - df: DataFrame containing 'hours_per_day' and 'music_effects' columns.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='hours_per_day', y='music_effects', hue='music_effects', palette=colors_custom, s=100, alpha=0.7)
    plt.title('Relationship Between Hours Per Day Listening to Music and Music Effects')
    plt.xlabel('Hours Per Day')
    plt.ylabel('Music Effects')
    plt.legend(title='Music Effects', bbox_to_anchor=(1.05, 1), loc='upper left')
    save_plot(os.path.join(output_dir, "hr_per_day_music_effects.png"))


def label_encode(df):
    """
    Perform label encoding on specified categorical columns of the DataFrame.

    Parameters:
    df (pd.DataFrame): The dataset containing categorical variables.
    categorical_columns (list): List of column names to be label encoded.

    Returns:
    pd.DataFrame: DataFrame with label encoded columns.
    """
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

    le = LabelEncoder()
    for column in categorical_columns:
        df[column] = le.fit_transform(df[column])
    return df

def plot_correlation_matrix(df, output_dir):
    """
    Generate and save a heatmap of the correlation matrix for the given DataFrame.

    Parameters:
    df (pd.DataFrame): The dataset containing numerical variables.
    output_dir (str): The directory where the plot will be saved.
    """
    label_encode(df)
    # Compute the correlation matrix
    corr_matrix = df.corr()
    
    # Plot the heatmap
    plt.figure(figsize=(16, 14))
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap='coolwarm',
        center=0,
        linewidths=0.5,
        fmt='.2f',
        cbar_kws={'shrink': 0.75}
    )
    plt.title('Correlation Matrix')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, "correlation_matrix.png"))


# --------------------------------------------------------------
# Main Script Execution
# --------------------------------------------------------------


def main():
    # Define file paths
    data_filepath = "C:/Users/sirin/Desktop/Github projects/Music-and-Mental-Health/data/interim/music_v_mental_health_cleaned.csv"
    output_dir = (
        "C:/Users/sirin/Desktop/Github projects/Music-and-Mental-Health/reports/Figures"
    )

    # Create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load data
    df = load_data(data_filepath)

    # Generate and save plots
    plot_age_distribution(df, output_dir)
    plot_streaming_service(df, output_dir)
    plot_musicianship_disctribution(df, output_dir)     
    plot_music_engagement_disctribution(df, output_dir)
    plot_favorite_genre(df, output_dir)
    plot_while_working(df, output_dir)
    plot_hours_per_day(df, output_dir)
    plot_bpm_distribution(df, output_dir)
    plot_age_group_genre(df, output_dir)
    plot_genre_frequency_counts(df, output_dir)
    plot_mental_illnesses(df, output_dir)
    plot_music_effects_pie_chart(df, output_dir)
    plot_favorite_genre_and_music_effects(df, output_dir)
    plot_bpm_range_and_music_effects(df, output_dir)
    plot_hours_per_day_and_music_effects(df, output_dir)
    plot_instrumentalist_composer_and_music_effects(df, output_dir)
    plot_mental_health_and_music_effects(df, output_dir)
    create_mental_health_columns(df)
    mental_health_columns = ['has_anxiety', 'has_depression', 'has_insomnia', 'has_ocd']
    visualize_music_preferences_for_mental_illness(df, mental_health_columns, output_dir, 
                                                   music_pref_col='favorite_genre')
    visualize_percentage_of_mental_illness(df, mental_health_columns, output_dir)
    
    visualize_hours_vs_music_effects(df,output_dir)
    plot_correlation_matrix(df, output_dir)
    print("Plots have been successfully saved.")


if __name__ == "__main__":
    main()
