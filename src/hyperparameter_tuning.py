"""
Pokemon Battle Predictor - Hyperparameter Tuning
================================================
GridSearch optimization for LogisticRegression and XGBoost models.
"""

import numpy as np
import pandas as pd
from libraries import *
from constants import *


def optimize_logistic_regression(X_train, y_train, cv_folds=10):
    """
    Optimize Logistic Regression hyperparameters using GridSearchCV.
    
    Args:
        X_train: Training feature matrix
        y_train: Training labels
        cv_folds: Number of cross-validation folds (default: 10)
        
    Returns:
        tuple: (best_model, best_params, best_score, cv_results)
    """
    print("\n" + "="*70)
    print("LOGISTIC REGRESSION - HYPERPARAMETER OPTIMIZATION")
    print("="*70)
    
    cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED)
    
    # Create pipeline with scaling
    log_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(solver='saga', random_state=SEED, max_iter=5000))
    ])
    
    # Parameter grid
    log_param_grid = {
        'clf__penalty': ['l1', 'l2', 'elasticnet'],
        'clf__C': [0.01, 0.1, 0.5, 1.0, 2.0],
        'clf__class_weight': [None, 'balanced']
    }
    
    # Add l1_ratio for elasticnet
    param_combinations = []
    for penalty in log_param_grid['clf__penalty']:
        for C in log_param_grid['clf__C']:
            for class_weight in log_param_grid['clf__class_weight']:
                if penalty == 'elasticnet':
                    for l1_ratio in [0.3, 0.5, 0.7]:
                        param_combinations.append({
                            'clf__penalty': penalty,
                            'clf__C': C,
                            'clf__class_weight': class_weight,
                            'clf__l1_ratio': l1_ratio
                        })
                else:
                    param_combinations.append({
                        'clf__penalty': penalty,
                        'clf__C': C,
                        'clf__class_weight': class_weight
                    })
    
    print(f"\n✓ Testing {len(param_combinations)} parameter combinations")
    print(f"✓ Using {cv_folds}-fold stratified cross-validation")
    
    # GridSearch
    log_grid = GridSearchCV(
        log_pipeline,
        log_param_grid,
        cv=cv_strategy,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )
    
    print("\n✓ Starting GridSearch...")
    log_grid.fit(X_train, y_train)
    
    # Results
    print(f"\n✓ Optimization Complete!")
    print(f"  - Best parameters: {log_grid.best_params_}")
    print(f"  - Best CV score: {log_grid.best_score_:.4f}")
    print(f"  - Best estimator penalty: {log_grid.best_estimator_.named_steps['clf'].penalty}")
    
    # Get detailed results
    cv_results_df = pd.DataFrame(log_grid.cv_results_)
    cv_results_df = cv_results_df.sort_values('rank_test_score')
    
    print(f"\n✓ Top 5 Parameter Combinations:")
    cols_to_show = [col for col in cv_results_df.columns if 'param_' in col] + \
                   ['mean_test_score', 'std_test_score', 'rank_test_score']
    print(cv_results_df[cols_to_show].head(5).to_string(index=False))
    
    return log_grid.best_estimator_, log_grid.best_params_, log_grid.best_score_, cv_results_df


def optimize_xgboost(X_train, y_train, cv_folds=10):
    """
    Optimize XGBoost hyperparameters using GridSearchCV.
    
    Args:
        X_train: Training feature matrix
        y_train: Training labels
        cv_folds: Number of cross-validation folds (default: 10)
        
    Returns:
        tuple: (best_model, best_params, best_score, cv_results)
    """
    print("\n" + "="*70)
    print("XGBOOST - HYPERPARAMETER OPTIMIZATION")
    print("="*70)
    
    cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED)
    
    # Parameter grid
    xgb_param_grid = {
        'n_estimators': [600, 800, 1000],
        'max_depth': [2, 3, 4],
        'learning_rate': [0.02, 0.03, 0.04],
        'subsample': [0.6, 0.7, 0.8],
        'colsample_bytree': [0.6, 0.7, 0.8],
        'min_child_weight': [50, 60, 75],
        'gamma': [0.3, 0.5],
        'reg_lambda': [3, 5],
        'reg_alpha': [0.3, 0.4]
    }
    
    # Calculate total combinations
    total_combinations = np.prod([len(v) for v in xgb_param_grid.values()])
    
    print(f"\n✓ Full grid has {total_combinations} combinations")
    print(f"✓ Using {cv_folds}-fold stratified cross-validation")
    
    # For practical purposes, use a reduced grid for initial search
    xgb_param_grid_reduced = {
        'n_estimators': [800, 1000],
        'max_depth': [2, 3],
        'learning_rate': [0.02, 0.03],
        'subsample': [0.6, 0.7],
        'colsample_bytree': [0.7, 0.8],
        'min_child_weight': [60, 75],
        'gamma': [0.5],
        'reg_lambda': [5],
        'reg_alpha': [0.4]
    }
    
    reduced_combinations = np.prod([len(v) for v in xgb_param_grid_reduced.values()])
    print(f"✓ Testing reduced grid: {reduced_combinations} combinations")
    
    # GridSearch
    xgb_grid = GridSearchCV(
        XGBClassifier(random_state=SEED, eval_metric='logloss', tree_method='hist'),
        xgb_param_grid_reduced,
        cv=cv_strategy,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )
    
    print("\n✓ Starting GridSearch...")
    xgb_grid.fit(X_train, y_train)
    
    # Results
    print(f"\n✓ Optimization Complete!")
    print(f"  - Best parameters: {xgb_grid.best_params_}")
    print(f"  - Best CV score: {xgb_grid.best_score_:.4f}")
    
    # Get detailed results
    cv_results_df = pd.DataFrame(xgb_grid.cv_results_)
    cv_results_df = cv_results_df.sort_values('rank_test_score')
    
    print(f"\n✓ Top 5 Parameter Combinations:")
    cols_to_show = [col for col in cv_results_df.columns if 'param_' in col] + \
                   ['mean_test_score', 'std_test_score', 'rank_test_score']
    print(cv_results_df[cols_to_show].head(5).to_string(index=False))
    
    return xgb_grid.best_estimator_, xgb_grid.best_params_, xgb_grid.best_score_, cv_results_df


def compare_models(log_results_df, xgb_results_df):
    """
    Compare optimization results between models.
    
    Args:
        log_results_df: DataFrame with LogReg CV results
        xgb_results_df: DataFrame with XGBoost CV results
    """
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    
    # Best scores
    log_best = log_results_df.iloc[0]['mean_test_score']
    log_std = log_results_df.iloc[0]['std_test_score']
    xgb_best = xgb_results_df.iloc[0]['mean_test_score']
    xgb_std = xgb_results_df.iloc[0]['std_test_score']
    
    print(f"\n✓ Best Cross-Validation Scores:")
    print(f"  - Logistic Regression: {log_best:.4f} (+/- {log_std:.4f})")
    print(f"  - XGBoost:             {xgb_best:.4f} (+/- {xgb_std:.4f})")
    print(f"  - Difference:          {abs(log_best - xgb_best):.4f}")
    
    if log_best > xgb_best:
        print(f"\n✓ Logistic Regression performs better by {(log_best - xgb_best)*100:.2f}%")
    elif xgb_best > log_best:
        print(f"\n✓ XGBoost performs better by {(xgb_best - log_best)*100:.2f}%")
    else:
        print(f"\n✓ Models perform equally well")
    
    # Statistical significance (approximate)
    combined_std = np.sqrt(log_std**2 + xgb_std**2)
    z_score = abs(log_best - xgb_best) / combined_std
    
    print(f"\n✓ Z-score for difference: {z_score:.2f}")
    if z_score > 1.96:
        print("  → Difference is statistically significant (p < 0.05)")
    else:
        print("  → Difference is not statistically significant")


def save_optimization_results(log_results_df, xgb_results_df, log_params, xgb_params):
    """
    Save optimization results to CSV files.
    
    Args:
        log_results_df: DataFrame with LogReg CV results
        xgb_results_df: DataFrame with XGBoost CV results
        log_params: Best LogReg parameters
        xgb_params: Best XGBoost parameters
    """
    print("\n✓ Saving optimization results...")
    
    # Save CV results
    log_results_df.to_csv('logreg_gridsearch_results.csv', index=False)
    xgb_results_df.to_csv('xgboost_gridsearch_results.csv', index=False)
    
    # Save best parameters
    best_params = {
        'logistic_regression': log_params,
        'xgboost': xgb_params
    }
    
    with open('best_hyperparameters.txt', 'w') as f:
        f.write("="*70 + "\n")
        f.write("BEST HYPERPARAMETERS\n")
        f.write("="*70 + "\n\n")
        
        f.write("Logistic Regression:\n")
        for key, value in log_params.items():
            f.write(f"  - {key}: {value}\n")
        
        f.write("\nXGBoost:\n")
        for key, value in xgb_params.items():
            f.write(f"  - {key}: {value}\n")
    
    print("  ✓ logreg_gridsearch_results.csv")
    print("  ✓ xgboost_gridsearch_results.csv")
    print("  ✓ best_hyperparameters.txt")


def perform_hyperparameter_tuning(X_train_log, y_train, X_train_xgb, cv_folds=10):
    """
    Complete hyperparameter tuning pipeline for both models.
    
    Args:
        X_train_log: Training features for LogReg
        y_train: Training labels
        X_train_xgb: Training features for XGBoost
        cv_folds: Number of CV folds (default: 10)
        
    Returns:
        dict: Dictionary with optimized models and results
    """
    print("\n" + "="*70)
    print("GRID SEARCH & HYPERPARAMETER OPTIMIZATION")
    print("="*70)
    
    # Optimize Logistic Regression
    log_model, log_params, log_score, log_results = optimize_logistic_regression(
        X_train_log, y_train, cv_folds
    )
    
    # Optimize XGBoost
    xgb_model, xgb_params, xgb_score, xgb_results = optimize_xgboost(
        X_train_xgb, y_train, cv_folds
    )
    
    # Compare models
    compare_models(log_results, xgb_results)
    
    # Save results
    save_optimization_results(log_results, xgb_results, log_params, xgb_params)
    
    print("\n" + "="*70)
    print("HYPERPARAMETER OPTIMIZATION COMPLETE")
    print("="*70)
    
    return {
        'log_model': log_model,
        'log_params': log_params,
        'log_score': log_score,
        'log_results': log_results,
        'xgb_model': xgb_model,
        'xgb_params': xgb_params,
        'xgb_score': xgb_score,
        'xgb_results': xgb_results
    }
