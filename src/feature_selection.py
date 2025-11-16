"""
Pokemon Battle Predictor - Feature Selection
============================================
Advanced feature selection with correlation pruning and GridSearch optimization.
"""

import numpy as np
import pandas as pd
from src.libraries import *
from src.constants import *


def remove_high_correlation_features(X_train, y_train, base_feature_cols, threshold=0.9):
    """
    Remove highly correlated features by keeping the one with higher importance.
    
    
    Args:
        X_train: Training feature matrix
        y_train: Training labels
        base_feature_cols: List of feature names
        threshold: Correlation threshold (default 0.9)
        
    Returns:
        tuple: (cleaned_feature_cols, high_corr_pairs)
    """
    
    
    print("\nStep 0: Removing highly correlated features (|corr| > 0.9)...")

    base_feature_cols = list(train_features.columns)
    X_features = train_df[base_feature_cols]
    correlation_matrix = X_features.corr()

    high_corr_threshold = 0.9
    features_to_remove = set()
    high_corr_pairs = []
    
    
    print("  Computing feature importance once...")
    temp_scaler = StandardScaler()
    X_temp_scaled = temp_scaler.fit_transform(X_features)
    temp_log = LogisticRegression(penalty='l2', solver='saga', C=0.1, 
                                 random_state=SEED, max_iter=1000)
    temp_log.fit(X_temp_scaled, y_train)
    temp_importance = dict(zip(base_feature_cols, np.abs(temp_log.coef_[0])))
    
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            if abs(correlation_matrix.iloc[i, j]) > high_corr_threshold:
                feat_1 = correlation_matrix.columns[i]
                feat_2 = correlation_matrix.columns[j]
                
                coef_1 = temp_importance.get(feat_1, 0)
                coef_2 = temp_importance.get(feat_2, 0)
                
                high_corr_pairs.append({
                    'feature_1': feat_1,
                    'feature_2': feat_2,
                    'correlation': correlation_matrix.iloc[i, j],
                    'removed': feat_2 if coef_1 >= coef_2 else feat_1,
                    'kept': feat_1 if coef_1 >= coef_2 else feat_2
                })
                
                if coef_1 >= coef_2:
                    features_to_remove.add(feat_2)
                    print(f"  Remove {feat_2} (corr={correlation_matrix.iloc[i, j]:.3f} with {feat_1})")
                else:
                    features_to_remove.add(feat_1)
                    print(f"  Remove {feat_1} (corr={correlation_matrix.iloc[i, j]:.3f} with {feat_2})")
    
    if high_corr_pairs:
        corr_report = pd.DataFrame(high_corr_pairs)
        corr_report.to_csv('high_correlation_report.csv', index=False)
        print(f"\n  High correlation report saved")
    
    base_feature_cols_cleaned = [f for f in base_feature_cols if f not in features_to_remove]
    print(f"\n  Removed {len(features_to_remove)} features due to high correlation")
    print(f"  Remaining: {len(base_feature_cols_cleaned)}/{len(base_feature_cols)} features")
    
    X_train = train_df[base_feature_cols_cleaned].fillna(-1.0)
    X_test = test_df[base_feature_cols_cleaned].fillna(-1.0)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    return base_feature_cols_cleaned, high_corr_pairs


def optimize_logreg_selector(X_train_scaled, y_train, base_feature_cols_cleaned):
    """
    Optimize LogisticRegression-based feature selection via GridSearch.
    .
    
    Args:
        X_train_scaled: Scaled training features
        y_train: Training labels
        base_feature_cols_cleaned: List of feature names after correlation removal
        
    Returns:
        tuple: (best_params, selected_features)
    """
    print("\nStep 1: L1/L2/ElasticNet-based SelectFromModel (LogReg)")
    
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    
    log_selector_param_grid = {
        'penalty': ['l1', 'l2', 'elasticnet'], 
        'C': [0.1, 0.2],  
        'threshold': ['mean', 'median', '0.75*mean'] 
    }
    
    
    print("  Pre-computing feature selections...")
    feature_selections = {}
    
    for penalty in log_selector_param_grid['penalty']:
        for C in log_selector_param_grid['C']:
            for threshold in log_selector_param_grid['threshold']:
                try:
                    if penalty == 'elasticnet':
                        selector = SelectFromModel(
                            LogisticRegression(penalty='elasticnet', solver='saga', C=C, 
                                             l1_ratio=0.5, random_state=SEED, max_iter=5000),
                            threshold=threshold
                        )
                    else:
                        selector = SelectFromModel(
                            LogisticRegression(penalty=penalty, solver='saga', C=C, 
                                             random_state=SEED, max_iter=5000),
                            threshold=threshold
                        )
                    
                    selector.fit(X_train_scaled, y_train)
                    selected_features = [f for f, s in zip(base_feature_cols_cleaned, selector.get_support()) if s]
                    
                    if len(selected_features) >= 20:  
                        feature_selections[(penalty, C, threshold)] = selected_features
                
                except Exception as e:
                    continue
    
    print(f"  Generated {len(feature_selections)} valid feature sets")
    
    
    print("\n  Evaluating feature sets")
    cv_fast = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)  
    best_log_score = 0
    best_log_params = None
    best_log_features = None
    
    for (penalty, C, threshold), selected_features in feature_selections.items():
        try:
            
            if penalty == 'elasticnet':
                temp_pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('clf', LogisticRegression(penalty='elasticnet', solver='saga', C=0.1, 
                                              l1_ratio=0.5, random_state=SEED, max_iter=1000))
                ])
            else:
                temp_pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('clf', LogisticRegression(penalty=penalty, solver='saga', C=0.1, 
                                              random_state=SEED, max_iter=1000))
                ])
            
           
            scores = cross_val_score(temp_pipeline, X_train[selected_features], y_train, 
                                    cv=cv_fast, scoring='accuracy', n_jobs=-1)
            score = scores.mean()
            
            if score > best_log_score:
                best_log_score = score
                best_log_params = {'penalty': penalty, 'C': C, 'threshold': threshold}
                best_log_features = selected_features
                print(f"  ✓ New best: penalty={penalty}, C={C}, threshold={threshold}, "
                      f"n_features={len(selected_features)}, CV={score:.4f}")
        
        except Exception as e:
            continue
    
    l2_selected = best_log_features
    print(f"\n  Best LogReg: {best_log_params}")
    print(f"  Selected: {len(l2_selected)}/{len(base_feature_cols_cleaned)} features")
        
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
        print("\nStep 2: XGBoost-based SelectFromModel - GridSearch")
    
    xgb_selector_param_grid = {
        'n_estimators': [600, 800],
        'max_depth': [2, 3],
        'threshold': ['mean', 'median', '0.5*mean', '0.75*mean']
    }
    
    best_xgb_score = 0
    best_xgb_params = None
    
    for n_est in xgb_selector_param_grid['n_estimators']:
        for max_d in xgb_selector_param_grid['max_depth']:
            for threshold in xgb_selector_param_grid['threshold']:
                try:
                    selector = SelectFromModel(
                        XGBClassifier(n_estimators=n_est, max_depth=max_d, random_state=SEED, 
                                    eval_metric='logloss', n_jobs=-1),
                        threshold=threshold
                    )
                    selector.fit(X_train, y_train)
                    selected_features = [f for f, s in zip(base_feature_cols_cleaned, selector.get_support()) if s]
                    
                    if len(selected_features) < 10:
                        continue
                    
                    temp_model = XGBClassifier(n_estimators=200, max_depth=3, random_state=SEED, 
                                             eval_metric='logloss', n_jobs=-1)
                    scores = cross_val_score(temp_model, X_train[selected_features], y_train, 
                                            cv=cv_strategy, scoring='accuracy', n_jobs=-1)
                    score = scores.mean()
                    
                    if score > best_xgb_score:
                        best_xgb_score = score
                        best_xgb_params = {'n_estimators': n_est, 'max_depth': max_d, 'threshold': threshold}
                        print(f"   New best: n_est={n_est}, max_depth={max_d}, threshold={threshold}, "
                              f"n_features={len(selected_features)}, CV={score:.4f}")
                
                except Exception as e:
                    continue
    
    xgb_selector = SelectFromModel(
        XGBClassifier(n_estimators=best_xgb_params['n_estimators'], 
                      max_depth=best_xgb_params['max_depth'], 
                      random_state=SEED, eval_metric='logloss', n_jobs=-1),
        threshold=best_xgb_params['threshold']
    )
    xgb_selector.fit(X_train, y_train)
    xgb_selected = [f for f, s in zip(base_feature_cols_cleaned, xgb_selector.get_support()) if s]
    print(f"\n  Best XGBoost: {best_xgb_params}")
    print(f"  Selected: {len(xgb_selected)}/{len(base_feature_cols_cleaned)} features")
        
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
