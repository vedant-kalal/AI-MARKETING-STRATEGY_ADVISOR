import os
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, confusion_matrix

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier


models = {
    "DecisionTree": DecisionTreeClassifier(class_weight='balanced', random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1),
    "ExtraTrees": ExtraTreesClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1),
    "KNN": KNeighborsClassifier(n_neighbors=2, n_jobs=-1)
}



# use path of processed data using Pathlib
proc_path = Path(r"C:\Github Projects\AI-MARKETING-STRATEGY_ADVISOR\data\processed\bank_modeling_ready.csv")
print(f"Loading processed data from: {proc_path}")
p_df = pd.read_csv(proc_path)

def train_test_split_data(p_df):
    X = p_df.drop('y', axis=1)
    y = p_df['y']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_eval(model):
    X_train, X_test, y_train, y_test = train_test_split_data(p_df)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', pos_label=1) # last value is '_' which we don't need
    accuracy = model.score(X_test, y_test)
    # average='binary':- Tells sklearn it’s a binary classification (two classes: yes/no, 0/1)
    # pos_label=1:- Specifies which class is considered the "positive" class for calculating precision and recall. Here, 1 means “yes” (subscribed)
    print(f"Precision: {p:.3f}  Recall: {r:.3f}  F1: {f1:.3f}")
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    return accuracy,p, r, f1

def display(models):
    print("Training and Evaluating Models...\n")
    model_accuracy = []
    for name, model in models.items():
        print(f"--------- {name} ---------")
        accuracy, p, r, f1 = train_eval(model)
        model_accuracy.append((name, accuracy, p, r, f1))
        print("\n")
    print("-"*30)
    print("HIGHEST Model Accuracy")
    print("-"*30)
    
    # Choose the best model by F1 score 
    best_model = max(model_accuracy, key=lambda x: x[4])
    print(f"Best Model: {best_model[0]} with Accuracy: {best_model[1]:.5f}, Precision: {best_model[2]:.5f}, Recall: {best_model[3]:.5f}, F1 Score: {best_model[4]:.5f}")
    return best_model

if __name__ == "__main__":
    display(models)