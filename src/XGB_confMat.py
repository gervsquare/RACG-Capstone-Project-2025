from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
from sklearn.feature_selection import RFECV
from xgboost import XGBClassifier, XGBRegressor
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def XGBoost_confusion_matrix(X_train, X_test, y_train, y_test):
        # For classification
    estimator = xgb.XGBClassifier(objective='binary:logistic', random_state=42, use_label_encoder=False, eval_metric='logloss')

    # Set up cross-validation
    # Adjust n_splits as needed
    cv = StratifiedKFold(n_splits=15, shuffle=True, random_state=42) 

    # Set up RFECV
    selector = RFECV (
        estimator=estimator,
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

    final_model = estimator
    final_model.fit(X_train_selected, y_train)

    # Evaluate the model
    predictions = final_model.predict(X_test_selected)
    accuracy = accuracy_score(y_test, predictions)
    print(f"\nModel Accuracy with XGBooster's selected features: {accuracy} or {accuracy * 100:.2f}%")
    print()

    ################################################################################################################################################
    # Calculate the confusion matrix
    cm = confusion_matrix(y_test, predictions)

    # Display the confusion matrix using ConfusionMatrixDisplay
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Without PCOS [0]', 'With PCOS [1]'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix for XGBooster Model")
    plt.show()

    # Calculate ROC curve values: FPR, TPR, and thresholds
    fpr, tpr, thresholds = roc_curve(y_test, predictions)

    # Calculate the AUC score
    roc_auc = auc(fpr, tpr)
    # Alternatively: roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f'\nROC AUC Score: {roc_auc:.4f}')

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='No Skill (AUC = 0.5)') #
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR) / Recall') #
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
