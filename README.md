# 🎧 Music Effects on Mental Health

## Project Overview

This project delves into the impact of music preferences on mental health through a comprehensive data analysis and modeling approach. By analyzing a dataset that includes listener demographics, listening habits, age, and mental health experiences, the goal is to uncover meaningful insights into how music influences mental well-being. Ultimately, the project aims to develop a classification model to predict the effects of music on new inputs.

## Key Components

### 🛠️ Data Preparation

Data preparation is handled by the `make_dataset.py` script, which performs data cleaning and transformation to ready the dataset for analysis and modeling. Cleaned and processed data is available in the `data` folder.

### 📊 Data Visualization

Effective visualization is essential for interpreting data. The `visualize.py` script generates a variety of exploratory and result-oriented visualizations. The resulting figures are saved in the `reports/figures` directory.

### 🔍 Exploratory Data Analysis (EDA) and Reporting

Detailed reports on exploratory data analysis and data preparation are available in the `reports` directory. These reports provide insights into the data and document the preparation and analysis steps.

### 🤖 Model Training and Prediction

- **Model Training**: Use the `train_model.py` script to train machine learning models with the prepared dataset.
- **Model Prediction**: The `predict_model.py` script makes predictions based on the trained models.

### 📚 Documentation

Find comprehensive project documentation, including methodology, data sources, and other relevant details, in the `docs` directory.

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


## 🗃️ Data Organization

The project's data is organized within the `data` folder:
- **Raw Data**: The original, unprocessed data.
- **Interim Data**: Transformed data used as an intermediate step.
- **Processed Data**: Final datasets prepared for analysis and modeling.

