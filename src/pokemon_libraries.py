"""
Pokemon Battle Predictor - Library Imports
==========================================
Centralized import statements for all dependencies.
"""

# Core data processing
import numpy as np
import pandas as pd
from pathlib import Path
from math import log
import warnings

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning - Model selection
from sklearn.model_selection import (
    StratifiedKFold, 
    cross_val_score, 
    GridSearchCV,
    learning_curve
)

# Machine learning - Preprocessing
from sklearn.preprocessing import StandardScaler

# Machine learning - Models
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# Machine learning - Evaluation
from sklearn.metrics import accuracy_score, log_loss

# Machine learning - Feature selection
from sklearn.feature_selection import (
    SelectFromModel,
    VarianceThreshold
)
from sklearn.inspection import permutation_importance

# Utilities
from itertools import product

# Configure warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
