import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

def grid(X_train, X_test, y_train, y_test):
    param_grid = {
        'n_estimators': [100, 200, 300], # Number of trees in the forest
        'max_depth': [None, 10, 20, 30], # Maximum depth of the tree
        'min_samples_split': [2, 5, 10], # Minimum number of samples required to split an internal node
        'criterion': ['gini', 'entropy'] # Function to measure the quality of a split
    }

    # Initialize the Random Forest Classifier
    rf = RandomForestClassifier(random_state=42)

    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,            # Number of cross-validation folds
        scoring='accuracy', # Metric to optimize (e.g., accuracy, precision, recall)
        verbose=2,       # Output logs during the process
        n_jobs=-1        # Use all available CPU cores for parallel processing
    )

    # Fit the grid search to the training data
    grid_search.fit(X_train, y_train)


    # Print the best hyperparameters found
    print(f"Best parameters: {grid_search.best_params_}")

    # Use the best estimator (model with optimal hyperparameters)
    best_model = grid_search.best_estimator_

    # Evaluate the best model on the test data
    accuracy = best_model.score(X_test, y_test)
    print(f"Test accuracy of the best model: {accuracy}")