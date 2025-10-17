import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import pandas as pd
from joblib import dump, parallel_backend
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from model_selection import models, display, p_df

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier


RANDOM_STATE = 42
MODELS_DIR = Path(r"C:\Github Projects\AI-MARKETING-STRATEGY_ADVISOR\models")
def fresh_estimator(name):
    if name == "DecisionTree": 
        return DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE)
    if name == "RandomForest": 
        return RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1)
    if name == "ExtraTrees":   
        return ExtraTreesClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1)
    if name == "KNN":          
        return KNeighborsClassifier()
    raise ValueError(name)

def space(name):
    if name == "DecisionTree":
        return {
            "max_depth":         (4, 24),
            "min_samples_split": (2, 20),
            "min_samples_leaf":  (1, 10),
            "criterion":         ["gini", "entropy"],
        }
    if name == "RandomForest":
        return {
            "n_estimators":      (120, 300),
            "max_depth":         (6, 24),
            "min_samples_split": (2, 20),
            "min_samples_leaf":  (1, 10),
            "max_features":      ["sqrt", "log2"],
            "bootstrap":         [True, False],
        }
    if name == "ExtraTrees":
        return {
            "n_estimators":      (150, 800),
            "max_depth":         (6, 24),
            "max_features":      ["sqrt", "log2"],
        }
    if name == "KNN":
        return {
            "n_neighbors":       (3, 40),
            "weights":           ["uniform", "distance"],
            "leaf_size":         (15, 50),
            "p":                 [1, 2],
        }

def main():
    # Split data
    X = p_df.drop(columns=["y"])
    y = p_df["y"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    # Get best base model name (display returns a tuple: (name, accuracy, precision, recall, f1))
    best_result = display(models)

    best_name = best_result[0]
    est = fresh_estimator(best_name)
    spc = space(best_name)

    # Randomized search
    tuner = RandomizedSearchCV(
        estimator=est,
        param_distributions=spc,
        scoring="f1",
        n_iter=25,
        random_state=RANDOM_STATE
    )

    print(f"Randomized tuning : {best_name}")

    # Fitting the tuner
    tuner.fit(Xtr, ytr)

    # Evaluate on test data directly
    yp = tuner.best_estimator_.predict(Xte)
    acc = accuracy_score(yte, yp)
    f1 = f1_score(yte, yp)
    rec = recall_score(yte, yp)
    pre = precision_score(yte, yp, zero_division=0)

    # Display results
    print("=== Tuned Model (No CV) ===")
    print(f"Model: {best_name}")
    print(f"Best Params: {tuner.best_params_}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {pre:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")


if __name__ == "__main__":
    main()


### Normal model withour tuning has already 98 percent accuracy.