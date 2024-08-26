import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import (
    RandomForestClassifier,
    StackingClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    VotingClassifier,
)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
import joblib
import os
from sklearn.base import clone


# ---------------------------------------------------------------------
# -------------------- Functions Definitions --------------------------
# ---------------------------------------------------------------------
# Define pipeline with feature selection and scaling
def create_pipeline(model):
    """
    Sets up a pipeline with feature selection and a model.

    Parameters:
    - model: A classifier object to be used in the pipeline.

    Returns:
    - Pipeline: A scikit-learn pipeline with feature selection and the provided model.
    """
    return Pipeline(
        [
            (
                "select_k_best",
                SelectKBest(score_func=f_classif, k=32),
            ),  # Combine multiple feature selection methods
            ("model", model),
        ]
    )


# Function to evaluate a model using cross-validation
def evaluate_model_cv(model, X_train, y_train, skf):
    """
    Performs cross-validation and computes the average accuracy.

    Parameters:
    - model: A machine learning classifier to be evaluated.
    - X_train: Training features.
    - y_train: Training target values.
    - skf: StratifiedKFold object for cross-validation.

    Returns:
    - float: The average accuracy across the cross-validation folds.
    """
    cv_accuracies = []
    for train_idx, test_idx in skf.split(X_train, y_train):
        X_cv_train, X_cv_test = X_train.iloc[train_idx], X_train.iloc[test_idx]
        y_cv_train, y_cv_test = y_train.iloc[train_idx], y_train.iloc[test_idx]

        # Clone the model to ensure independence of folds
        clf = clone(model)

        # Train the model
        clf.fit(X_cv_train, y_cv_train)

        # Evaluate on the CV test set
        y_cv_pred = clf.predict(X_cv_test)
        cv_accuracy = accuracy_score(y_cv_test, y_cv_pred)
        cv_accuracies.append(cv_accuracy)

    # Return average cross-validation accuracy
    return sum(cv_accuracies) / len(cv_accuracies)


# Function to evaluate a model on validation set
def evaluate_model(model, X_train, y_train, X_val, y_val):
    """
    Trains and evaluates a model on the validation set.

    Parameters:
    - model: A machine learning classifier to be evaluated.
    - X_train: Training features.
    - y_train: Training target values.
    - X_val: Validation features.
    - y_val: Validation target values.

    Returns:
    - tuple: A tuple containing validation accuracy, classification report, and predicted labels on the validation set.
    """
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    class_report = classification_report(y_val, y_val_pred)

    return val_accuracy, class_report, y_val_pred


# Function to save and print model evaluation results
def save_and_print_evaluation_results(
    models, X_train, y_train, X_val, y_val, skf, output_file
):
    """
    Evaluates multiple models using cross-validation and validation set, and saves the results to a file.

    Parameters:
    - models: List of tuples containing models and their names.
    - X_train: Training features.
    - y_train: Training target values.
    - X_val: Validation features.
    - y_val: Validation target values.
    - skf: StratifiedKFold object for cross-validation.
    - output_file: Path to the file where the results will be saved.
    """
    with open(output_file, "w") as f:
        for model, name in models:
            f.write(f"Evaluating {name}...\n")
            print(f"Evaluating {name}...")

            # Cross-validation
            avg_cv_accuracy = evaluate_model_cv(model, X_train, y_train, skf)
            f.write(
                f"{name} average cross-validation accuracy: {avg_cv_accuracy:.4f}\n"
            )
            print(f"{name} average cross-validation accuracy: {avg_cv_accuracy:.4f}")

            # Validation
            (
                val_accuracy,
                class_report,
                y_val_pred
            ) = evaluate_model(model, X_train, y_train, X_val, y_val)
            f.write(f"{name} validation accuracy: {val_accuracy:.4f}\n")
            f.write(f"{name} classification report:\n{class_report}\n")
            f.write("-" * 50 + "\n")
            print(f"{name} validation accuracy: {val_accuracy:.4f}")
            print(f"{name} classification report:\n{class_report}")
            print("-" * 50)


def print_not_selected_features(stacking_model, X_train_val):
    """
    Prints the features that are not selected by the feature selector in each estimator of the stacking model.

    Parameters:
    - stacking_model: A StackingClassifier object with feature selectors.
    - X_train_val: Features from the combined training and validation set.
    """
    for name, estimator in stacking_model.named_estimators_.items():
        if hasattr(estimator.named_steps["select_k_best"], "get_support"):
            selected_mask = estimator.named_steps["select_k_best"].get_support()
            not_selected_features = X_train_val.columns[~selected_mask]
            print(f"Features not selected by {name}:")
            print(not_selected_features)  # freq_metal and ocd


def plot_normalized_confusion_matrix(y_true, y_pred, output_dir):
    """
    Plot and save a normalized confusion matrix.

    Parameters:
    - y_true: True labels of the test set.
    - y_pred: Predicted labels from the model.
    - output_dir: Directory where the plots will be saved.
    """

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Plot Normalized Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 7))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=np.unique(y_true),
        yticklabels=np.unique(y_true),
    )
    plt.title("Normalized Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(f"{output_dir}/normalized_confusion_matrix.png")
    plt.close()


# ---------------------------------------------------------------------
# -------------------- Main Code Execution ----------------------------
# ---------------------------------------------------------------------

# Load data
data_filepath = "../../data/processed/music_v_mental_health_processed.csv"
df = pd.read_csv(data_filepath)

# Split data into Features and Target
target_column = "music_effects"
X = df.drop(columns=[target_column])
y = df[target_column]

# Initial split to get train+val and test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Further split train+val into train and val
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
)  # 0.25 * 0.8 = 0.2

# Train model using training data
# Define models
models = [
    (RandomForestClassifier(random_state=42), "RandomForest"),
    (SVC(random_state=42), "SVC"),
    (KNeighborsClassifier(), "KNeighbors"),
    (
        LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42),
        "LogisticRegression",
    ),
    (
        GaussianNB(),
        "GaussianNB",
    ),
    (
        StackingClassifier(
            estimators=[
                ("rf", RandomForestClassifier(random_state=42)),
                ("svc", SVC(random_state=42, probability=True)),
                ("knn", KNeighborsClassifier()),
            ],
            final_estimator=LogisticRegression(),
        ),
        "StackingClassifier",
    ),
    (
        BaggingClassifier(
            estimator=RandomForestClassifier(random_state=42), random_state=42
        ),
        "BaggingClassifier",
    ),
    (GradientBoostingClassifier(random_state=42), "GradientBoostingClassifier"),
    (
        AdaBoostClassifier(
            estimator=RandomForestClassifier(random_state=42),
            algorithm="SAMME",
            random_state=42,
        ),
        "AdaBoostClassifier",
    ),
    (
        VotingClassifier(
            estimators=[
                ("rf", RandomForestClassifier(random_state=42)),
                ("svc", SVC(random_state=42, probability=True)),
                ("gnb", GaussianNB()),
            ],
            voting="soft",
        ),
        "VotingClassifier",
    ),
]

# StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
output_file = "../../reports/model_evaluation_results.txt"
# Cross-validation and evaluation
save_and_print_evaluation_results(
    models, X_train, y_train, X_val, y_val, skf, output_file
)

# Define the stacking model
stacking_model = StackingClassifier(
    estimators=[
        (
            "rf",
            create_pipeline(
                RandomForestClassifier(
                    random_state=42,
                )
            ),
        ),
        (
            "svc",
            create_pipeline(
                SVC(C=1.0, kernel="linear", probability=True, random_state=42)
            ),
        ),
        ("knn", create_pipeline(KNeighborsClassifier())),
    ],
    final_estimator=LogisticRegression(
        solver="lbfgs", C=1.0, penalty="l2", max_iter=1000, random_state=42
    ),
)

# Evaluate the Stacking Model
# Train the stacking model on the combined training and validation data and
# Evaluate on the test data
test_accuracy, class_report, y_test_pred = evaluate_model(
    stacking_model, X_train_val, y_train_val, X_test, y_test
)
# Calculate training accuracy
y_train_pred = stacking_model.predict(X_train_val)
train_accuracy = accuracy_score(y_train_val, y_train_pred)


print(f"StackingClassifier Training accuracy: {train_accuracy:.4f}")
print(f"StackingClassifier Test accuracy: {test_accuracy:.4f}")

# Retrieve and display the features not selected by SelectKBest
print_not_selected_features(stacking_model, X_train_val)

# Print the classification report
print(f"StackingClassifier classification report:\n{class_report}")


# Plot normalized Confusion Matrix
output_dir = "../../reports/Figures"
plot_normalized_confusion_matrix(y_test, y_test_pred, output_dir)


# Save the trained model
model_filepath = "../../models/music_effects_model.pkl"
joblib.dump(stacking_model, model_filepath)

print(f"Model training complete. Trained model saved to '{model_filepath}'.")
