import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier,
    StackingClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    VotingClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
import joblib
import os
from sklearn.base import clone


data_filepath = "C:/Users/sirin/Desktop/Github projects/Music-and-Mental-Health/data/processed/music_v_mental_health_processed.csv"
model_filepath = "C:/Users/sirin/Desktop/Github projects/Music-and-Mental-Health/models/music_effects_model.pkl"
X_test_filepath = "C:/Users/sirin/Desktop/Github projects/Music-and-Mental-Health/data/processed/X_test.csv"
y_test_filepath = "C:/Users/sirin/Desktop/Github projects/Music-and-Mental-Health/data/processed/y_test.csv"
output_dir = (
    "C:/Users/sirin/Desktop/Github projects/Music-and-Mental-Health/reports/Figures"
)
# Load data
df = pd.read_csv(data_filepath)

# Split data into training, validation, and test sets
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

# Save the test data for later use
X_test.to_csv(X_test_filepath, index=False)
y_test.to_csv(y_test_filepath, index=False)

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
    # Adding BaggingClassifier
    (
        BaggingClassifier(
            estimator=RandomForestClassifier(random_state=42), random_state=42
        ),
        "BaggingClassifier",
    ),
    # Adding GradientBoostingClassifier
    (GradientBoostingClassifier(random_state=42), "GradientBoostingClassifier"),
    # Adding AdaBoostClassifier
    (
        AdaBoostClassifier(
            estimator=RandomForestClassifier(random_state=42),
            algorithm="SAMME",
            random_state=42,
        ),
        "AdaBoostClassifier",
    ),
    # Adding VotingClassifier
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

# Cross-validation and evaluation
# Open the file to save the results
with open("../../reports/model_evaluation_results.txt", "w") as f:
    # Cross-validation and evaluation
    for model, name in models:
        f.write(f"Evaluating {name}...\n")
        print(f"Evaluating {name}...")
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

        # Average CV accuracy
        avg_cv_accuracy = sum(cv_accuracies) / len(cv_accuracies)
        f.write(f"{name} average cross-validation accuracy: {avg_cv_accuracy:.4f}\n")
        print(f"{name} average cross-validation accuracy: {avg_cv_accuracy:.4f}")

        # Final evaluation on the validation set
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        train_accuracy = accuracy_score(y_train, y_train_pred)

        f.write(f"{name} Training accuracy: {train_accuracy:.4f}\n")
        f.write(f"{name} validation accuracy: {val_accuracy:.4f}\n")
        f.write(
            f"{name} classification report:\n{classification_report(y_val, y_val_pred)}\n"
        )
        f.write("-" * 50 + "\n")
        print(f"{name} validation accuracy: {val_accuracy:.4f}")
        print(
            f"{name} classification report:\n{classification_report(y_val, y_val_pred)}"
        )
        print("-" * 50)

from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFE


# Define pipeline with feature selection and scaling
def create_pipeline(model):
    return Pipeline(
        [
            (
                "rfe",
                RFE(
                    estimator=RandomForestClassifier(random_state=42),
                    n_features_to_select=23,
                ),
            ),  # Combine multiple feature selection methods
            ("model", model),
        ]
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
        ("svc", create_pipeline(SVC(C=1.0, 
                                    kernel="linear", 
                                    random_state=42))),
        ("knn", create_pipeline(KNeighborsClassifier())),
    ],
    final_estimator=LogisticRegression(solver="lbfgs", 
                                       C=1.0, 
                                       penalty="l2", 
                                       max_iter=1000, 
                                       random_state=42
    ),
)

# Train the stacking model on the combined training and validation data
stacking_model.fit(X_train_val, y_train_val)

# Evaluate the model on the test data
y_test_pred = stacking_model.predict(X_test)
y_train_pred = stacking_model.predict(X_train_val)
# Calculate accuracy
test_accuracy = accuracy_score(y_test, y_test_pred)
train_accuracy = accuracy_score(y_train_val, y_train_pred)
print(f"StackingClassifier Training accuracy: {train_accuracy:.4f}")

print(f"StackingClassifier Test accuracy: {test_accuracy:.4f}")

# Print the classification report
print(
    f"StackingClassifier classification report:\n{classification_report(y_test, y_test_pred)}"
)






# Evaluate model using validation data
"""Evaluates the model on the validation data."""
conf_matrix = confusion_matrix(y_test, y_test_pred)
class_report = classification_report(y_test, y_test_pred, output_dict=True)
y_pred_proba = model.predict_proba(y_test)

# Plot enhanced visualizations
class_names = sorted(y_test.unique())

# def plot_normalized_confusion_matrix(cm, class_names, output_dir):
cm_normalized = conf_matrix.astype("float") / conf_matrix.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm_normalized,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Validation Confusion Matrix (Normalized)")
plt.savefig(
    os.path.join(output_dir, "RandomForest_validation_confusion_matrix.png"),
    bbox_inches="tight",
)
plt.close()
plt.show()


# def plot_precision_recall_f1(y_true, y_scores, class_names, output_dir):
plt.figure(figsize=(14, 7))

for i, class_name in enumerate(class_names):
    precision, recall, _ = precision_recall_curve(y_test == i, y_pred_proba[:, i])
    plt.plot(recall, precision, label=f"{class_name} (Precision-Recall)")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend(loc="best")
plt.grid(True)
plt.savefig(
    os.path.join(output_dir, "RandomForest_validation_precision_recall_curve.png"),
    bbox_inches="tight",
)
plt.close()
plt.show()


# def plot_roc_curve(y_true, y_scores, class_names, output_dir):
plt.figure(figsize=(14, 7))

for i, class_name in enumerate(class_names):
    fpr, tpr, _ = roc_curve(y_test == i, y_pred_proba[:, i])
    plt.plot(fpr, tpr, label=f"{class_name} (ROC curve)")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="best")
plt.grid(True)
plt.savefig(
    os.path.join(output_dir, "RandomForest_validation_roc_curve.png"),
    bbox_inches="tight",
)
plt.close()
plt.show()












# Save the trained model
joblib.dump(model, model_filepath)

print(f"Model training complete. Trained model saved to '{model_filepath}'.")
print(f"Test data saved to '{X_test_filepath}' and '{y_test_filepath}'.")
