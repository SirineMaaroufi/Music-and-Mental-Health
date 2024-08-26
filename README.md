# 🎧 Music Effects on Mental Health

## Project Overview

This project investigates the relationship between music listening habits and mental health, motivated by a personal passion for music and its positive impact on well-being. As an avid listener of various music genres, I have experienced firsthand how different types of music can influence mood and emotional states. This project aims to explore these effects more systematically, focusing particularly on individuals with mental health conditions.

By analyzing a comprehensive dataset that includes listener habits, demographics, and self-reported mental health experiences, we seek to uncover the impact of different music genres on mental well-being. \
The ultimate goal is to develop a predictive model that can assess the potential effects of music on mental health.


## Key Components

## 🗃️ Data Organization
Data is organized in the [`data`](data) folder:
- **Raw Data**: Original, unprocessed data.
- **Interim Data**: Transformed data used as an intermediate step.
- **Processed Data**: Final datasets prepared for analysis and modeling.

For a detailed description of the data, refer to the [`Data Description`](docs/Data%20Description.md) file.

## 🛠️ Data Preparation
Data preparation is crucial for accurate analysis and modeling. The [`make_dataset.py`](src/data/make_dataset.py) script handles data cleaning and transformation. The cleaned data is stored in the [`data/interim`](data/interim) directory. Details are documented in the [`Data Preparation Report`](reports/Data%20Preparation%20Report.md).

## 📊 Data Visualization
Effective visualization aids in data interpretation. The [`visualize.py`](src/visualization/visualize.py) script creates exploratory and result-oriented visualizations. Documentation for the script is available in the [`Data Visualization Script Documentation`](docs/Data%20Visualization%20Script%20Documentation.md). Figures are saved in the [`reports/figures`](reports/Figures/) directory.

## 🔍 Exploratory Data Analysis (EDA) and Data Analysis
The [`EDA and Data Analysis Report`](reports/EDA%20&%20Data%20Analysis%20Report.md) provides detailed insights into the dataset, highlighting key trends, correlations, and findings that guide preprocessing decisions.

## 🔧 Feature Engineering

Feature engineering enhances model performance. The [`build_feature.py`](src/features/build_features.py) script transforms raw data into features that better represent underlying patterns. Detailed steps are documented in the [`Feature Engineering Documentation`](docs/Feature%20Engineering%20Documentation.md). Preprocessed data is stored in the [`data/processed`](data/processed/) directory.

## 🤖 Model Training and Prediction

- **Model Training**: Various machine learning models were evaluated, with the StackingClassifier, which combines RandomForest, SVC, KNeighborsClassifier, and Logistic Regression, achieving a **0.90 Test Accuracy**. Training and evaluation are handled by the [`train_model.py`](src/models/train_model.py) script, and the trained model is saved in the [`models`](models) directory. See the [`Modeling Report`](reports/Modeling%20Report.md) for details.

- **Model Prediction**: The [`predict_model.py`](src/models/predict_model.py) script makes predictions based on the trained model.

## 📚 Documentation

Find comprehensive project documentation, including methodology, data descriptions, and sources, in the [`docs`](docs) directory.

## 🛠️ Tools and Technologies

- **Python** 🐍: For data preparation, cleaning, visualization, and modeling.
- **Libraries**:
  - [pandas](https://pandas.pydata.org/) 📈: Data manipulation and analysis.
  - [NumPy](https://numpy.org/) 🔢: Numerical operations and data handling.
  - [seaborn](https://seaborn.pydata.org/) 🌈: Statistical data visualization.
  - [matplotlib](https://matplotlib.org/) 📊: Plotting and visualization.
  - [scikit-learn](https://scikit-learn.org/) ⚙️: Machine learning and modeling.

## 🖥️ Development Environment
- **Code Editor**: [Visual Studio Code (VS Code)](https://code.visualstudio.com/) 

## 📈🔍 Outcomes and Results


### 💡🔎 **Analysis Insights**
___
* **Summary**: Music listening has a generally positive impact on mental health, with notable benefits for those who listen daily. Rock is the most popular genre among younger individuals.
* **Key Takeaways**:
  - **Demographics and Music Preferences:** 🎸 Younger individuals (ages 14-27) favor Rock, Pop, and Metal, with Rock being the most popular genre.
  - **Listening Habits and Mental Health:** 🎧 Daily music listening (1 to 3.5 hours) generally has a positive impact on mental health, reducing anxiety and depression.
  - **Musical Background and Engagement:** 🎤 Many respondents are musicians or actively engage with music, but there’s no significant difference in mental health impact between musicians and non-musicians.
  - **Correlation Analysis:**
    - **Music Effects:** 🎵 Music preferences correlate positively with favorite genres and negatively with work activities.
    - **Mental Health Conditions:** 🧠 Strong correlations among mental health conditions suggest a tendency for co-occurrence.
    - **Music Characteristics:** 🎶 Tempo and listening duration affect mental health, though not exclusively.

  **Overall Impact:** 🌟 Music positively influences mental well-being, with personalized music therapy showing promise for further benefits.

### 🤖📊 **Modeling Results**
___
* **Summary**: The StackingClassifier achieved a high test accuracy, indicating strong model performance. Precision, recall, and F1-scores were well-balanced.
* **Key Takeaways:**
  - **StackingClassifier Performance:** 🏆 The StackingClassifier, combining RandomForest, SVC, KNeighborsClassifier, and Logistic Regression, achieved a **0.90 Test Accuracy**.
  - **Precision and Recall:** 🎯 The model demonstrated excellent precision and recall, particularly for the worsening class (-1) with perfect scores.
  - **Balanced F1-Scores:** 📈 F1-scores were balanced across all classes, indicating robust model performance.

## 🔮 Future Work

1. **Broadening Data Scope:** Expanding dataset to include diverse demographics and additional variables like listening context and physiological responses.

2. **Enhancing Models:** Exploring advanced algorithms, optimizing hyperparameters, and improving feature selection for better prediction accuracy.

3. **Longitudinal Studies:** Conducting studies to assess long-term effects of music on mental health and tracking changes over time.
