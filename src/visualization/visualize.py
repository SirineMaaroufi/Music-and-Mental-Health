"""
This script performs data visualization on the cleaned Music and Mental Health dataset.

Author: Sirine Maaroufi
Date: August 2024
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

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


def plot_category_distribution(df, output_dir):
    """
    Generate and save a bar plot showing the proportion of responses across various categories.

    Parameters:
    df (pd.DataFrame): The dataset containing categorical response data.
    output_dir (str): The directory where the plot will be saved.
    """
    # Melt the DataFrame to long format
    df_melted = df.melt(
        value_vars=["instrumentalist", "composer", "exploratory", "foreign_languages"],
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
        "Proportion of Responses Across Various Categories with Percentage Breakdown"
    )
    plt.xlabel("Category")
    plt.ylabel("Percentage")
    plt.legend(title="Response")
    save_plot(os.path.join(output_dir, "category_distribution.png"))


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
    plt.xticks(rotation=45)
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


def plot_genre_frequency(df, output_dir):
    """
    Generate and save a count plot showing the frequency of listening to each genre, ordered by frequency categories.

    Parameters:
    df (pd.DataFrame): The dataset containing genre frequency data.
    output_dir (str): The directory where the plot will be saved.
    """
    # Melt the DataFrame to long format for easier plotting

    df_melt = df.melt(
        id_vars=["favorite_genre"],
        value_vars=[col for col in df.columns if "freq" in col],
        var_name="Genre",
        value_name="Frequency",
    )

    # Define the order of frequency categories

    frequency_order = ["Very frequently", "Sometimes", "Rarely", "Never"]
    df_melt["Frequency"] = pd.Categorical(
        df_melt["Frequency"], categories=frequency_order, ordered=True
    )
    # Calculate the total count of each genre

    genre_total_counts = (
        df_melt.groupby("favorite_genre")["Frequency"]
        .count()
        .reset_index(name="TotalCount")
    )
    # Order genres by their total counts

    ordered_genres = genre_total_counts.sort_values(by="TotalCount", ascending=False)[
        "favorite_genre"
    ].tolist()
    df_melt["favorite_genre"] = pd.Categorical(
        df_melt["favorite_genre"], categories=ordered_genres, ordered=True
    )
    # Create and save the count plot

    plt.figure(figsize=(14, 8))
    sns.countplot(
        data=df_melt,
        y="favorite_genre",
        hue="Frequency",
        palette="tab20",
        order=ordered_genres,
    )
    plt.title("Frequency of Listening to Each Genre Ordered by Frequency Categories")
    plt.xlabel("Count")
    plt.ylabel("Genre")
    plt.legend(title="Frequency", bbox_to_anchor=(1.05, 1), loc="upper left")
    save_plot(os.path.join(output_dir, "genre_frequency.png"))


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


# --------------------------------------------------------------
# Main Script Execution
# --------------------------------------------------------------


def main():
    # Define file paths
    data_filepath = "C:/Users/sirin/Desktop/Github projects/Music-and-Mental-Health/data/processed/music_v_mental_health_cleaned.csv"
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
    plot_category_distribution(df, output_dir)
    plot_favorite_genre(df, output_dir)
    plot_while_working(df, output_dir)
    plot_hours_per_day(df, output_dir)
    plot_bpm_distribution(df, output_dir)
    plot_age_group_genre(df, output_dir)
    plot_genre_frequency(df, output_dir)
    plot_mental_illnesses(df, output_dir)
    plot_music_effects_pie_chart(df, output_dir)
    plot_favorite_genre_and_music_effects(df, output_dir)
    plot_bpm_range_and_music_effects(df, output_dir)
    plot_hours_per_day_and_music_effects(df, output_dir)
    plot_instrumentalist_composer_and_music_effects(df, output_dir)
    plot_mental_health_and_music_effects(df, output_dir)

    print("Plots have been successfully saved.")


if __name__ == "__main__":
    main()
