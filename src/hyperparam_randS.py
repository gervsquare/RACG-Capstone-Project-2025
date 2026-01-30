import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split

def rand(X_train, X_test, y_train, y_test):
    rfc = RandomForestClassifier(random_state=42)

    # Define the parameter distributions
    param_dist = {
        'n_estimators': np.arange(50, 300, 50),
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    random_search = RandomizedSearchCV(
        estimator=rfc,
        param_distributions=param_dist,
        n_iter=50,
        scoring='accuracy',
        cv=5,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    random_search.fit(X_train, y_train)

    # Print the best hyperparameters found
    print(f"Best parameters: {random_search.best_params_}")

    # Use the best estimator (model with optimal hyperparameters)
    best_model1 = random_search.best_estimator_

    # Evaluate the best model on the test data
    accuracy1 = best_model1.score(X_test, y_test)
    print(f"Test accuracy of the best model: {accuracy1}")