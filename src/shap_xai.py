import numpy as np
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from catboost import CatBoostClassifier

def shap_process(xtrain, ytrain, xtest, ytest):
    # Create a SHAP explainer
    # The library automatically selects the best explainer (e.g., TreeExplainer for tree models)
    xg = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
    xg.fit(xtrain, ytrain)

    explainer = shap.Explainer(xg)

    # Calculate SHAP values for the test data
    shap_values = explainer(xtest)

    ####################################################3
    print(shap_values.shape)
    print(f"Shape of X: {xtrain.shape}") 
    # check the index you are trying to access
    print("\n Beeswarm SHAP Plot")
    shap.plots.beeswarm(shap_values)
    print("\n Summary SHAP Plot")
    shap.summary_plot(shap_values, xtest, plot_type="bar")
    print("\n Waterfall SHAP Plot")
    shap.plots.waterfall(shap_values[0])