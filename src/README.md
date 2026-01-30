# RACG Capstone Project – Healthcare: Predict patient readmission or disease likelihood of PCOS
## Given that I was also diagnosed of PCOS for 10 years, I have sought to have this area as my chosen Capstone focus which will provide at least an initial aid or a “sneak peek” for women who are undiagnosed and have yet to get themselves checked. This is also for health care providers or insurers to utilize as this is an AI-driven classification that can shift the model from reactive treatment (expensive) to proactive management (cost-effective).

# Installation
## Below are all the libraries that I had to manually install inside the main.py to be able to use it for some functions and methods. Make sure that when you utilize this source code, you have the following installed:
```bash
pip install pandas
pip install "numpy<2"
pip install numpy
pip install -U scikit-learn 
pip install --upgrade scikit-learn
pip install shap
pip install xgboost catboost
pip install shap matplotlib
pip install imbalanced-learn
pip install fairlearn
pip install lightgbm
```
## Remember that when you install these, you need to restart the kernel you are utilizing. From when I created this project, I was using VSCode IDE, with the kernel base(Python 3.12.7).

## In addition to the installation of python libraries, make sure within your local unit, you have the following installed:
### - Visual Studio Code
### - Git [https://git-scm.com/] - install the appropriate version for your OS

# Data Source
## Source of data is from the PCOS Diagnosis Kaggle page [https://www.kaggle.com/code/karnikakapoor/pcos-diagnosis/] by Karnika Kapoor

# Python Libraries
## Providing the list of ALL libraries that are needed to be included in this project for it to run
### main.py
```bash
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from prePro_dropCol import drop_col #This is to allow you to call the function in prePro_dropCol.py
from prePro_conVal import convert_val #This is to allow you to call the function in prePro_conVal.py

from sklearn.impute import SimpleImputer

import shap
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier

from collections import Counter
from sklearn.datasets import make_classification
from imblearn.combine import SMOTEENN

from modelSelect import modSel
from shap_xai import shap_process
from feat import featureSelect
from hyperparam_gridS import grid #This is to allow you to call the function in hyperparam_gridS.py
from hyperparam_randS import rand #This is to allow you to call the function in hyperparam_randS.py

import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
import re
from sklearn.metrics import accuracy_score

from RFC_confMat import RFClassifier_confusion_matrix #This is to allow you to call the function in RFC_confMat.py

from xgboost import XGBClassifier, XGBRegressor
import xgboost as xgb
```

### for the Python libraries inside the .py files, kindly go through them, they are declared at the very first instance.

# Report a Bug
## Describe the bug
### A clear and concise description of what the bug is.

### Steps to reproduce
### Steps to reproduce the behavior.

### Expected behavior
### A clear and concise description of what you expected to happen.

### Environment
###  - OS: [e.g. Arch Linux]
###  - Other details that you think may affect.

### Additional context
### Add any other context here.

# Contact
## For any concerns, kindly reach out to me via my email - robigervi@outlook.com

