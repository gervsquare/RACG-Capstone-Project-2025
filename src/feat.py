import pandas as pd
import numpy as np
from sklearn.feature_selection import RFECV
from xgboost import XGBClassifier, XGBRegressor
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score

def featureSelect(X_train, X_test, y_train, y_test):
# For classification
    estimator = {
        "XGBoost": xgb.XGBClassifier(objective='binary:logistic', random_state=42, use_label_encoder=False, eval_metric='logloss'),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    # Train and evaluate each model
    for name, model in estimator.items():
        # Set up cross-validation
        # Adjust n_splits as needed
        cv = StratifiedKFold(n_splits=15, shuffle=True, random_state=42) 

        # Set up RFECV
        selector = RFECV (
            estimator=model,
            step=1, # Remove 1 feature at each iteration
            cv=cv,
            scoring='accuracy', # Choose appropriate scoring metric
            n_jobs=-1, # Use all available cores
            #min_features_to_select=1, # Minimum number of features to select
            #verbose=0
        )

        # Fit RFECV
        selector = selector.fit(X_train, y_train)

        print(f"Optimal number of features: {selector.n_features_}")
        print(f"Best cross-validation score (accuracy): {selector.cv_results_['mean_test_score'].max():.4f}")
        print(f"Selected features mask: {selector.support_}")

        # Transform training and testing data to include only selected features
        X_train_selected = selector.transform(X_train)
        X_test_selected = selector.transform(X_test)

        final_model = model
        final_model.fit(X_train_selected, y_train)

        # Evaluate the model
        predictions = final_model.predict(X_test_selected)
        accuracy = accuracy_score(y_test, predictions)
        print(f"\nModel Accuracy with {name}'s selected features: {accuracy} or {accuracy * 100:.2f}%")
        print()