"""
Pokemon Battle Predictor - Feature Selection
============================================
Advanced feature selection with correlation pruning and GridSearch optimization.
"""

import numpy as np
import pandas as pd
from libraries import *
from constants import *


def remove_high_correlation_features(X_train, y_train, base_feature_cols, threshold=0.9):
    """
    Remove highly correlated features by keeping the one with higher importance.
    OPTIMIZED VERSION: Uses faster correlation calculation and vectorized operations.
    
    Args:
        X_train: Training feature matrix
        y_train: Training labels
        base_feature_cols: List of feature names
        threshold: Correlation threshold (default 0.9)
        
    Returns:
        tuple: (cleaned_feature_cols, high_corr_pairs)
    """
    print(f"\n✓ Step 0: Removing highly correlated features (|corr| > {threshold})...")
    
    X_features = X_train[base_feature_cols]
    
    # Faster correlation calculation using numpy
    print("  - Computing correlation matrix...")
    correlation_matrix = X_features.corr().abs()  # Use absolute values directly
    
    # Get upper triangle to avoid duplicates
    upper_tri = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )
    
    # Find features with correlation > threshold
    to_drop_pairs = [
        (column, row, correlation_matrix.loc[row, column])
        for column in upper_tri.columns
        for row in upper_tri.index
        if upper_tri.loc[row, column] > threshold
    ]
    
    if not to_drop_pairs:
        print("  ✓ No highly correlated features found")
        return base_feature_cols, []
    
    print(f"  - Found {len(to_drop_pairs)} correlated pairs")
    
    # Quick importance estimation using simple logistic regression
    print("  - Estimating feature importances (fast)...")
    temp_scaler = StandardScaler()
    X_temp_scaled = temp_scaler.fit_transform(X_features)
    
    # Use simpler model for faster fitting
    temp_log = LogisticRegression(
        penalty='l2', 
        solver='saga', 
        C=0.1, 
        random_state=SEED, 
        max_iter=500,  # Reduced from 1000
        tol=1e-3  # Slightly relaxed tolerance for faster convergence
    )
    temp_log.fit(X_temp_scaled, y_train)
    temp_importance = dict(zip(base_feature_cols, np.abs(temp_log.coef_[0])))
    
    # Decide which features to remove
    features_to_remove = set()
    high_corr_pairs = []
    
    for feat_1, feat_2, corr_val in to_drop_pairs:
        coef_1 = temp_importance.get(feat_1, 0)
        coef_2 = temp_importance.get(feat_2, 0)
        
        # Keep feature with higher importance
        if coef_1 >= coef_2:
            removed = feat_2
            kept = feat_1
        else:
            removed = feat_1
            kept = feat_2
        
        if removed not in features_to_remove:
            high_corr_pairs.append({
                'feature_1': feat_1,
                'feature_2': feat_2,
                'correlation': corr_val,
                'removed': removed,
                'kept': kept
            })
            features_to_remove.add(removed)
            print(f"  Remove {removed} (corr={corr_val:.3f} with {kept})")
    
    # Save report
    if high_corr_pairs:
        corr_report = pd.DataFrame(high_corr_pairs)
        corr_report.to_csv('high_correlation_report.csv', index=False)
        print(f"\n  ✓ High correlation report saved: high_correlation_report.csv")
    
    base_feature_cols_cleaned = [f for f in base_feature_cols if f not in features_to_remove]
    print(f"\n  ✓ Removed {len(features_to_remove)} features due to high correlation")
    print(f"  ✓ Remaining: {len(base_feature_cols_cleaned)}/{len(base_feature_cols)} features")
    
    return base_feature_cols_cleaned, high_corr_pairs


def optimize_logreg_selector(X_train_scaled, y_train, base_feature_cols_cleaned):
    """
    Optimize LogisticRegression-based feature selection via GridSearch.
    OPTIMIZED VERSION: Reduced search space and faster CV.
    
    Args:
        X_train_scaled: Scaled training features
        y_train: Training labels
        base_feature_cols_cleaned: List of feature names after correlation removal
        
    Returns:
        tuple: (best_params, selected_features)
    """
    print("\n✓ Step 1: L1/L2/ElasticNet-based SelectFromModel (LogReg) - GridSearch")
    
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)  # Reduced from 10 to 5
    
    # OPTIMIZED: Reduced parameter grid
    param_grid = {
        'penalty': ['l1', 'l2', 'elasticnet'],
        'C': [0.05, 0.1, 0.2],  # 3 values (was could be more)
        'threshold': ['mean', '0.75*mean', '1.25*mean']  # 3 values (reduced from 5)
    }
    
    best_score = 0
    best_params = None
    best_features = None
    
    total_combinations = len(param_grid['penalty']) * len(param_grid['C']) * len(param_grid['threshold'])
    print(f"  - Testing {total_combinations} combinations (optimized grid)")
    
    current = 0
    for penalty in param_grid['penalty']:
        for C in param_grid['C']:
            for threshold in param_grid['threshold']:
                current += 1
                try:
                    # Add l1_ratio for elasticnet
                    if penalty == 'elasticnet':
                        selector = SelectFromModel(
                            LogisticRegression(
                                penalty='elasticnet', 
                                solver='saga', 
                                C=C, 
                                l1_ratio=0.5,  # Fixed at 0.5 for speed
                                random_state=SEED, 
                                max_iter=3000,  # Reduced from 5000
                                tol=1e-3
                            ),
                            threshold=threshold
                        )
                    else:
                        selector = SelectFromModel(
                            LogisticRegression(
                                penalty=penalty, 
                                solver='saga', 
                                C=C, 
                                random_state=SEED, 
                                max_iter=3000,  # Reduced from 5000
                                tol=1e-3
                            ),
                            threshold=threshold
                        )
                    
                    selector.fit(X_train_scaled, y_train)
                    selected_features = [f for f, s in zip(base_feature_cols_cleaned, selector.get_support()) if s]
                    
                    if len(selected_features) < 10:
                        continue
                    
                    # Quick CV score with fewer iterations
                    if penalty == 'elasticnet':
                        temp_pipeline = Pipeline([
                            ('scaler', StandardScaler()),
                            ('clf', LogisticRegression(
                                penalty='elasticnet', 
                                solver='saga', 
                                C=0.1, 
                                l1_ratio=0.5, 
                                random_state=SEED, 
                                max_iter=500,  # Reduced from 1000
                                tol=1e-3
                            ))
                        ])
                    else:
                        temp_pipeline = Pipeline([
                            ('scaler', StandardScaler()),
                            ('clf', LogisticRegression(
                                penalty=penalty, 
                                solver='saga', 
                                C=0.1, 
                                random_state=SEED, 
                                max_iter=500,  # Reduced from 1000
                                tol=1e-3
                            ))
                        ])
                    
                    # Use DataFrame indexing for selected features
                    X_selected = pd.DataFrame(X_train_scaled, columns=base_feature_cols_cleaned)[selected_features]
                    scores = cross_val_score(temp_pipeline, X_selected, y_train, 
                                            cv=cv_strategy, scoring='accuracy', n_jobs=-1)
                    score = scores.mean()
                    
                    if score > best_score:
                        best_score = score
                        best_params = {'penalty': penalty, 'C': C, 'threshold': threshold}
                        best_features = selected_features
                        print(f"  ✓ [{current}/{total_combinations}] New best: penalty={penalty}, C={C}, "
                              f"threshold={threshold}, n_features={len(selected_features)}, CV={score:.4f}")
                
                except Exception as e:
                    continue
    
    print(f"\n  ✓ Best LogReg params: {best_params}")
    print(f"  ✓ Selected: {len(best_features)}/{len(base_feature_cols_cleaned)} features")
    
    return best_params, best_features


def optimize_xgb_selector(X_train, y_train, base_feature_cols_cleaned):
    """
    Optimize XGBoost-based feature selection via GridSearch.
    
    Args:
        X_train: Training feature matrix
        y_train: Training labels
        base_feature_cols_cleaned: List of feature names after correlation removal
        
    Returns:
        tuple: (best_params, selected_features)
    """
    print("\n✓ Step 2: XGBoost-based SelectFromModel - GridSearch")
    
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    param_grid = {
        'n_estimators': [600, 800],
        'max_depth': [2, 3],
        'threshold': ['mean', 'median', '0.5*mean', '0.75*mean']
    }
    
    best_score = 0
    best_params = None
    best_features = None
    
    for n_est in param_grid['n_estimators']:
        for max_d in param_grid['max_depth']:
            for threshold in param_grid['threshold']:
                try:
                    selector = SelectFromModel(
                        XGBClassifier(n_estimators=n_est, max_depth=max_d, 
                                    random_state=SEED, eval_metric='logloss'),
                        threshold=threshold
                    )
                    selector.fit(X_train, y_train)
                    selected_features = [f for f, s in zip(base_feature_cols_cleaned, selector.get_support()) if s]
                    
                    if len(selected_features) < 10:
                        continue
                    
                    temp_model = XGBClassifier(n_estimators=200, max_depth=3, 
                                             random_state=SEED, eval_metric='logloss')
                    scores = cross_val_score(temp_model, X_train[selected_features], y_train, 
                                           cv=cv_strategy, scoring='accuracy', n_jobs=-1)
                    score = scores.mean()
                    
                    if score > best_score:
                        best_score = score
                        best_params = {'n_estimators': n_est, 'max_depth': max_d, 'threshold': threshold}
                        best_features = selected_features
                        print(f"  ✓ New best: n_est={n_est}, max_depth={max_d}, threshold={threshold}, "
                              f"n_features={len(selected_features)}, CV={score:.4f}")
                
                except Exception as e:
                    continue
    
    print(f"\n  ✓ Best XGBoost params: {best_params}")
    print(f"  ✓ Selected: {len(best_features)}/{len(base_feature_cols_cleaned)} features")
    
    return best_params, best_features


def perform_feature_selection(X_train_df, y_train, base_feature_cols):
    """
    Perform complete feature selection pipeline.
    
    Args:
        X_train_df: Training DataFrame with all features
        y_train: Training labels
        base_feature_cols: List of all feature names
        
    Returns:
        dict: Dictionary with selected features for each model
    """
    print("\n" + "="*70)
    print("FEATURE SELECTION & PRUNING WITH GRIDSEARCH")
    print("="*70)
    
    # Step 0: Remove highly correlated features
    base_feature_cols_cleaned, _ = remove_high_correlation_features(
        X_train_df, y_train, base_feature_cols, threshold=0.9
    )
    
    X_train = X_train_df[base_feature_cols_cleaned].fillna(-1.0)
    
    # Scale features for LogReg
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Step 1: Optimize LogReg selector
    log_params, log_selected = optimize_logreg_selector(
        X_train_scaled, y_train, base_feature_cols_cleaned
    )
    
    # Step 2: Optimize XGBoost selector
    xgb_params, xgb_selected = optimize_xgb_selector(
        X_train, y_train, base_feature_cols_cleaned
    )
    
    # Step 3: Union of selected features
    selected_base_features = list(set(log_selected) | set(xgb_selected))
    
    print(f"\n✓ Final feature set: {len(selected_base_features)} features")
    print(f"  - LogReg features: {len(log_selected)}")
    print(f"  - XGBoost features: {len(xgb_selected)}")
    print(f"  - Union: {len(selected_base_features)}")
    
    return {
        'all_features': base_feature_cols_cleaned,
        'log_features': log_selected,
        'xgb_features': xgb_selected,
        'union_features': selected_base_features,
        'log_params': log_params,
        'xgb_params': xgb_params
    }
