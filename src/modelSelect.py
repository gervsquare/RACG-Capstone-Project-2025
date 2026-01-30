import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from catboost import CatBoostClassifier

def modSel (xtrain, ytrain, xtest, ytest):
    # Dictionary to store models
    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "SVM": SVC(random_state=42),
        "XGBoost": xgb.XGBClassifier(eval_metric='logloss', random_state=42),
        "CatBoost": CatBoostClassifier(verbose=0, random_state=42) # verbose=0 to suppress CatBoost output
    }

    # Train and evaluate each model
    for name, model in models.items():
        # Train the model
        model.fit(xtrain, ytrain)
        
        # Make predictions on the test set
        y_pred = model.predict(xtest)
        
        # Calculate and print accuracy
        accuracy = accuracy_score(ytest, y_pred)
        print(f"{name} Accuracy: {accuracy * 100:.2f}%")
        mse = mean_squared_error(ytest, y_pred)
        print((f"{name} Mean Squared Error: {mse * 100:.2f}%")) # the close the mse is, the better
        print()