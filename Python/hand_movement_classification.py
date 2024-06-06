"""
Author: Shohada Sharmin
Date: 2024-06-05
Description: This script is designed to classify hand movements (reaching and grasping) from video data that has been processed into CSV files.
             Each CSV file contains coordinates of body parts over time, and the script extracts features like movement, velocity, acceleration, finger spread, and hand angle.
             The models are trained and evaluated on this feature set, with results saved for further analysis.
             
             The script performs the following tasks:
             1. Loads and preprocesses data from each CSV file.
             2. Performs feature engineering, including the calculation of finger spread and hand angle.
             3. Trains machine learning models (Logistic Regression, Random Forest, Gradient Boosting, SVM) and an ensemble model (Voting Classifier).
             4. Evaluates the models using metrics such as accuracy, precision, recall, and F1-score.
             5. Saves the trained models and the preprocessing transformers (imputer, scaler).
             6. Loads the trained models and makes predictions on test data.
             7. Performs hyperparameter tuning using GridSearchCV.
             8. Calculates and prints average metrics across all scenarios.

Dependencies:
- sys: Used for system-specific parameters and functions.
- pandas: Used for data manipulation and analysis.
- sklearn: Used for machine learning models, preprocessing, and evaluation metrics.
- joblib: Used for saving and loading models.
- numpy: Used for numerical operations.
- os: Used for file and directory operations.
- re: Used for regular expression operations.
"""

# Function to calculate finger spread based on distances between finger tips
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.impute import SimpleImputer
from joblib import dump, load
import numpy as np
import os
import re

# Function to calculate finger spread based on distances between finger tips
def calculate_finger_spread(data):
    finger_tips = ['T1', 'T2', 'T3', 'T4', 'T5']
    distances = []
    for i in range(len(finger_tips)):
        for j in range(i + 1, len(finger_tips)):
            distances.append(np.sqrt((data[f'{finger_tips[i]}_x'] - data[f'{finger_tips[j]}_x'])**2 + 
                                     (data[f'{finger_tips[i]}_y'] - data[f'{finger_tips[j]}_y'])**2))
    data['finger_spread'] = np.mean(distances, axis=0)
    return data

# Function to calculate hand angle based on wrist and base part coordinates
def calculate_hand_angle(data, wrist_part='W1', base_part='B1'):
    data['hand_angle'] = np.arctan2(data[f'{wrist_part}_y'] - data[f'{base_part}_y'],
                                    data[f'{wrist_part}_x'] - data[f'{base_part}_x']) * 180 / np.pi
    return data

# Function to perform feature engineering on the dataset
def feature_engineering(data):
    body_parts = ['SL', 'SR', 'EL', 'W1', 'W2', 'B1', 'B2', 'B3', 'B4', 'B5', 'T1', 'T2', 'T3', 'T4', 'T5']
    for part in body_parts:
        for coord in ['x', 'y']:
            data[f'{part}_{coord}_movement'] = data[f'{part}_{coord}'].diff().fillna(0)
            data[f'{part}_{coord}_velocity'] = data[f'{part}_{coord}_movement'] / (1/59.94)
            data[f'{part}_{coord}_acceleration'] = data[f'{part}_{coord}_velocity'].diff().fillna(0) / (1/59.94)
    data = calculate_finger_spread(data)
    data = calculate_hand_angle(data)
    return data

# Function to save the trained model
def save_model(model, model_name, feature_names):
    model_path = f'D:/Research/2024/2finalprep/classify/{model_name}.joblib'
    dump({'model': model, 'features': feature_names}, model_path)
    print(f'Model saved to {model_path}')

# Function to save the imputer and scaler transformers
def save_transformers(imputer, scaler, model_name):
    imputer_path = f'D:/Research/2024/2finalprep/classify/{model_name}_imputer.joblib'
    scaler_path = f'D:/Research/2024/2finalprep/classify/{model_name}_scaler.joblib'
    try:
        dump(imputer, imputer_path)
        dump(scaler, scaler_path)
        print(f'Imputer and Scaler saved successfully to {imputer_path} and {scaler_path}.')
    except Exception as e:
        print(f"Failed to save transformers: {str(e)}")

# Function to load the trained model and make predictions on test data
def load_and_predict(model_filename, X, y, scenario, output_file):
    model_path = os.path.join('D:/Research/2024/2finalprep/classify', model_filename)
    imputer_path = os.path.join('D:/Research/2024/2finalprep/classify', f'{scenario}_imputer.joblib')
    scaler_path = os.path.join('D:/Research/2024/2finalprep/classify', f'{scenario}_scaler.joblib')

    model_data = load(model_path)
    model = model_data['model']
    feature_names = model_data['features']
    
    imputer = load(imputer_path)
    scaler = load(scaler_path)

    X = pd.DataFrame(imputer.transform(X.drop('Label', axis=1)), columns=X.drop('Label', axis=1).columns)
    X = pd.DataFrame(scaler.transform(X), columns=X.columns)
    X = X[feature_names]  # Ensure the features in X are in the same order as during training

    y_pred = model.predict(X)

    # Individual model predictions
    individual_models = {
        'Logistic Regression': model.named_estimators_['lr'],
        'Random Forest': model.named_estimators_['rf'],
        'Gradient Boosting': model.named_estimators_['gb'],
        'SVM': model.named_estimators_['svm']
    }

    for name, mdl in individual_models.items():
        y_pred_ind = mdl.predict(X)
        accuracy = accuracy_score(y, y_pred_ind)
        precision_v = precision_score(y, y_pred_ind, pos_label=1, zero_division=0)
        recall_v = recall_score(y, y_pred_ind, pos_label=1, zero_division=0)
        f1_v = f1_score(y, y_pred_ind, pos_label=1, zero_division=0)
        precision_nv = precision_score(y, y_pred_ind, pos_label=0, zero_division=0)
        recall_nv = recall_score(y, y_pred_ind, pos_label=0, zero_division=0)
        f1_nv = f1_score(y, y_pred_ind, pos_label=0, zero_division=0)

        print(f"{name} Accuracy: {accuracy}", file=output_file)
        print(f"{name} Precision (V): {precision_v}", file=output_file)
        print(f"{name} Precision (NV): {precision_nv}", file=output_file)
        print(f"{name} Recall (V): {recall_v}", file=output_file)
        print(f"{name} Recall (NV): {recall_nv}", file=output_file)
        print(f"{name} F1-score (V): {f1_v}", file=output_file)
        print(f"{name} F1-score (NV): {f1_nv}", file=output_file)

    # Ensemble model predictions
    ensemble_accuracy = accuracy_score(y, y_pred)
    ensemble_precision_v = precision_score(y, y_pred, pos_label=1, zero_division=0)
    ensemble_precision_nv = precision_score(y, y_pred, pos_label=0, zero_division=0)
    ensemble_recall_v = recall_score(y, y_pred, pos_label=1, zero_division=0)
    ensemble_recall_nv = recall_score(y, y_pred, pos_label=0, zero_division=0)
    ensemble_f1_v = f1_score(y, y_pred, pos_label=1, zero_division=0)
    ensemble_f1_nv = f1_score(y, y_pred, pos_label=0, zero_division=0)

    print(f"\nVoting Classifier Accuracy: {ensemble_accuracy}", file=output_file)
    print(f"Voting Classifier Precision (V): {ensemble_precision_v}", file=output_file)
    print(f"Voting Classifier Precision (NV): {ensemble_precision_nv}", file=output_file)
    print(f"Voting Classifier Recall (V): {ensemble_recall_v}", file=output_file)
    print(f"Voting Classifier Recall (NV): {ensemble_recall_nv}", file=output_file)
    print(f"Voting Classifier F1-score (V): {ensemble_f1_v}", file=output_file)
    print(f"Voting Classifier F1-score (NV): {ensemble_f1_nv}", file=output_file)

# Function for hyperparameter tuning using GridSearchCV
def hyperparameter_tuning(model, param_grid, X_train, y_train):
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"Best parameters for {model.__class__.__name__}: {grid_search.best_params_}")
    return grid_search.best_estimator_

# Function to train and evaluate models
def train_and_evaluate(data, label_col, model_name):
    # Extract features and labels
    y = data[label_col].dropna()
    X = data.loc[y.index].drop(['Label', 'Reaching', 'Grasping'], axis=1)

    # Impute missing values and scale features
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    metrics = {
        'Logistic Regression': {'accuracy': [], 'precision_v': [], 'recall_v': [], 'f1_v': [], 'precision_nv': [], 'recall_nv': [], 'f1_nv': []},
        'Random Forest': {'accuracy': [], 'precision_v': [], 'recall_v': [], 'f1_v': [], 'precision_nv': [], 'recall_nv': [], 'f1_nv': []},
        'Gradient Boosting': {'accuracy': [], 'precision_v': [], 'recall_v': [], 'f1_v': [], 'precision_nv': [], 'recall_nv': [], 'f1_nv': []},
        'SVM': {'accuracy': [], 'precision_v': [], 'recall_v': [], 'f1_v': [], 'precision_nv': [], 'recall_nv': [], 'f1_nv': []},
        'Voting Classifier': {'accuracy': [], 'precision_v': [], 'recall_v': [], 'f1_v': [], 'precision_nv': [], 'recall_nv': [], 'f1_nv': []}
    }

    for train_index, test_index in skf.split(X_scaled, y):
        X_train, X_test = X_scaled.iloc[train_index], X_scaled.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Initialize models
        logreg = LogisticRegression(multi_class='multinomial', solver='saga', max_iter=10000, C=0.1)
        rf = RandomForestClassifier(random_state=42, n_estimators=50, max_depth=3)
        gb = GradientBoostingClassifier(random_state=42, n_estimators=50, learning_rate=0.01, max_depth=3)
        svm = SVC(probability=True, random_state=42, C=0.1, kernel='rbf')

        # Hyperparameter tuning
        param_grid_logreg = {'C': [0.01, 0.1, 1, 10]}
        param_grid_rf = {'n_estimators': [50, 100, 150], 'max_depth': [2, 3, 5]}
        param_grid_gb = {'n_estimators': [50, 100, 150], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [2, 3, 5]}
        param_grid_svm = {'C': [0.01, 0.1, 1, 10], 'kernel': ['linear', 'rbf']}

        logreg = hyperparameter_tuning(logreg, param_grid_logreg, X_train, y_train)
        rf = hyperparameter_tuning(rf, param_grid_rf, X_train, y_train)
        gb = hyperparameter_tuning(gb, param_grid_gb, X_train, y_train)
        svm = hyperparameter_tuning(svm, param_grid_svm, X_train, y_train)

        # Create an ensemble of the models
        ensemble = VotingClassifier(estimators=[('lr', logreg), ('rf', rf), ('gb', gb), ('svm', svm)],
                                    voting='soft', weights=[0.4, 0.3, 0.2, 0.1])

        models = {
            'Logistic Regression': logreg,
            'Random Forest': rf,
            'Gradient Boosting': gb,
            'SVM': svm,
            'Voting Classifier': ensemble
        }

        for name, model in models.items():
            # Train the model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision_v = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
            recall_v = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
            f1_v = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
            precision_nv = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
            recall_nv = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
            f1_nv = f1_score(y_test, y_pred, pos_label=0, zero_division=0)

            metrics[name]['accuracy'].append(accuracy)
            metrics[name]['precision_v'].append(precision_v)
            metrics[name]['recall_v'].append(recall_v)
            metrics[name]['f1_v'].append(f1_v)
            metrics[name]['precision_nv'].append(precision_nv)
            metrics[name]['recall_nv'].append(recall_nv)
            metrics[name]['f1_nv'].append(f1_nv)

    # Save the trained ensemble model and transformers
    save_model(ensemble, model_name, X_train.columns.tolist())
    save_transformers(imputer, scaler, model_name)

    return metrics

# Function to print average metrics
def print_average_metrics(metrics, output_file):
    for name, metric_values in metrics.items():
        mean_accuracy = np.mean(metric_values['accuracy'])
        mean_precision_v = np.mean(metric_values['precision_v'])
        mean_recall_v = np.mean(metric_values['recall_v'])
        mean_f1_v = np.mean(metric_values['f1_v'])
        mean_precision_nv = np.mean(metric_values['precision_nv'])
        mean_recall_nv = np.mean(metric_values['recall_nv'])
        mean_f1_nv = np.mean(metric_values['f1_nv'])

        print(f"avg. {name} Accuracy: {mean_accuracy}", file=output_file)
        print(f"avg. {name} Precision (V): {mean_precision_v}", file=output_file)
        print(f"avg. {name} Precision (NV): {mean_precision_nv}", file=output_file)
        print(f"avg. {name} Recall (V): {mean_recall_v}", file=output_file)
        print(f"avg. {name} Recall (NV): {mean_recall_nv}", file=output_file)
        print(f"avg. {name} F1-score (V): {mean_f1_v}", file=output_file)
        print(f"avg. {name} F1-score (NV): {mean_f1_nv}", file=output_file)

# Function to evaluate a specific scenario by training and testing models
def evaluate_scenario(train_files, test_files, scenario_id):
    # Load and preprocess training and testing data
    train_data = load_and_preprocess_data(train_files)
    test_data = load_and_preprocess_data(test_files)

    # Add binary labels for Reaching and Grasping
    train_data['Reaching'] = train_data['Label'].replace({1: 1, 2: 0, 3: None, 4: None}).astype('float')
    train_data['Grasping'] = train_data['Label'].replace({1: None, 2: None, 3: 1, 4: 0}).astype('float')

    # Evaluate for each scenario
    all_metrics = {'Reaching': {}, 'Grasping': {}}
    for scenario, label_col in {'Reaching': 'Reaching', 'Grasping': 'Grasping'}.items():
        metrics = train_and_evaluate(train_data, label_col, f"model_{scenario.lower()}_scenario_{scenario_id}")
        all_metrics[scenario] = metrics

    return all_metrics

# Function to load and preprocess data from CSV files
def load_and_preprocess_data(filepaths):
    data = pd.concat([feature_engineering(pd.read_csv(filepath)) for filepath in filepaths], ignore_index=True)
    return data

# List of all dataset files
all_files = [
    'D:/Research/2024/2finalprep/classify/P1_Cheerio_NVF.csv', 
    'D:/Research/2024/2finalprep/classify/P1_Cheerio_VF.csv', 
    'D:/Research/2024/2finalprep/classify/P2_Cheerio_NVF.csv', 
    'D:/Research/2024/2finalprep/classify/P2_Cheerio_VF.csv', 
    'D:/Research/2024/2finalprep/classify/P3_Cheerio_NVF.csv', 
    'D:/Research/2024/2finalprep/classify/P3_Cheerio_VF.csv', 
    'D:/Research/2024/2finalprep/classify/P4_Cheerio_NVF.csv', 
    'D:/Research/2024/2finalprep/classify/P4_Cheerio_VF.csv', 
    'D:/Research/2024/2finalprep/classify/P5_Cheerio_NVF.csv', 
    'D:/Research/2024/2finalprep/classify/P5_Cheerio_VF.csv', 
    'D:/Research/2024/2finalprep/classify/P6_Cheerio_NVF.csv', 
    'D:/Research/2024/2finalprep/classify/P6_Cheerio_VF.csv'
]

# Separate the vision and no-vision files
vision_files = [file for file in all_files if '_VF' in file]
no_vision_files = [file for file in all_files if '_NVF' in file]

# Create scenarios
all_metrics = {'Reaching': {}, 'Grasping': {}}
for vision_file, no_vision_file in zip(vision_files, no_vision_files):
    test_files = [vision_file, no_vision_file]
    train_files = [file for file in all_files if file not in test_files]
    participant_id = re.search(r'P\d+', vision_file).group()
    scenario_id = f"{participant_id}"
    scenario_metrics = evaluate_scenario(train_files, test_files, scenario_id)
    for scenario in ['Reaching', 'Grasping']:
        for model in scenario_metrics[scenario]:
            if model not in all_metrics[scenario]:
                all_metrics[scenario][model] = {'accuracy': [], 'precision_v': [], 'recall_v': [], 'f1_v': [], 'precision_nv': [], 'recall_nv': [], 'f1_nv': []}
            for metric in all_metrics[scenario][model]:
                all_metrics[scenario][model][metric].extend(scenario_metrics[scenario][model][metric])

# Print average metrics for all grasping and all reaching scenarios
with open('avg_grasping_metrics.txt', 'w') as grasping_output_file, open('avg_reaching_metrics.txt', 'w') as reaching_output_file:
    print("Average Grasping Metrics:\n", file=grasping_output_file)
    print_average_metrics(all_metrics['Grasping'], grasping_output_file)
    print("Average Reaching Metrics:\n", file=reaching_output_file)
    print_average_metrics(all_metrics['Reaching'], reaching_output_file)

print("All scenarios completed.")
