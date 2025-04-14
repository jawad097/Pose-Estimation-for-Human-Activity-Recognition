import pandas as pd
import numpy as np
import argparse
import os
import pickle
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_pose_data(csv_file):
    """
    Load pose data from CSV file
    
    Args:
        csv_file: Path to the CSV file
        
    Returns:
        df: Pandas DataFrame with pose data
    """
    try:
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} records from {csv_file}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def prepare_features(df):
    """
    Prepare features for model training
    
    Args:
        df: Pandas DataFrame with pose data
        
    Returns:
        X: Feature matrix
        y: Target labels
        feature_names: List of feature names
    """
    # Check if we have activity labels
    if 'activity' not in df.columns:
        print("Error: No activity labels found in the dataset")
        return None, None, None
    
    # Get feature columns (excluding timestamp, frame_id, and activity)
    feature_cols = [col for col in df.columns if col not in ['timestamp', 'frame_id', 'activity']]
    
    # Prioritize angle and ratio features if available
    angle_ratio_cols = [col for col in feature_cols if 'angle' in col or 'ratio' in col]
    if angle_ratio_cols:
        print(f"Using {len(angle_ratio_cols)} angle and ratio features")
        feature_cols = angle_ratio_cols
    else:
        print(f"Using {len(feature_cols)} features")
    
    # Prepare feature matrix and target labels
    X = df[feature_cols].fillna(0)  # Replace NaN with 0
    y = df['activity']
    
    return X, y, feature_cols

def train_activity_model(X, y, feature_names, output_dir=None):
    """
    Train a machine learning model for activity recognition
    
    Args:
        X: Feature matrix
        y: Target labels
        feature_names: List of feature names
        output_dir: Directory to save model and results (optional)
        
    Returns:
        model: Trained model
        scaler: Feature scaler
    """
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    print(f"Activity classes: {len(np.unique(y))}")
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model
    print("Training Random Forest model...")
    start_time = time.time()
    
    # Define parameter grid for grid search
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    
    # Create and train model with grid search
    model = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    
    model.fit(X_train_scaled, y_train)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Best parameters: {model.best_params_}")
    
    # Evaluate model on test set
    y_pred = model.predict(X_test_scaled)
    
    # Print classification report
    print("\nClassification Report:")
    report = classification_report(y_test, y_pred)
    print(report)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    importances = model.best_estimator_.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.title('Feature Importances')
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    
    # Save model, scaler, and results if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model and scaler
        with open(os.path.join(output_dir, 'activity_model.pkl'), 'wb') as f:
            pickle.dump(model, f)
        
        with open(os.path.join(output_dir, 'feature_scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)
        
        # Save feature names
        with open(os.path.join(output_dir, 'feature_names.pkl'), 'wb') as f:
            pickle.dump(feature_names, f)
        
        # Save classification report
        with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
            f.write("Classification Report:\n")
            f.write(report)
            f.write("\n\nBest Parameters:\n")
            f.write(str(model.best_params_))
        
        # Save confusion matrix plot
        plt.figure(0)
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
        
        # Save feature importance plot
        plt.figure(1)
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
        
        print(f"\nModel and results saved to {output_dir}")
    
    return model, scaler

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train activity recognition model from pose data')
    parser.add_argument('csv_file', type=str, help='Path to the CSV file with pose data')
    parser.add_argument('--output', '-o', type=str, default='model',
                        help='Directory to save model and results (default: model)')
    args = parser.parse_args()
    
    # Load data
    df = load_pose_data(args.csv_file)
    if df is None:
        return
    
    # Prepare features
    X, y, feature_names = prepare_features(df)
    if X is None:
        return
    
    # Train model
    model, scaler = train_activity_model(X, y, feature_names, args.output)
    
    print("\nModel training completed.")
    
    # Show plots if not saving to output directory
    if not args.output:
        plt.show()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
