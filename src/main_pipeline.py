"""
Pokemon Battle Predictor - Main Pipeline
========================================
Complete pipeline for training and predicting Pokemon battle outcomes.
"""

from src.libraries import *
from src.constants import *
from src.functions import load_data, clean_data, extract_all_features
from src.feature_selection import perform_feature_selection
from src.hyperparameter_tuning import perform_hyperparameter_tuning
from src.feature_importance import perform_feature_importance_analysis
from src.ensemble_optimization import (
    cross_validate_models, optimize_ensemble_weights, create_stacking_ensemble,
    compare_ensemble_methods, analyze_overfitting, create_submissions
)


def main():
    """Main execution pipeline."""
    
    print("="*70)
    print("POKEMON BATTLE PREDICTOR - GEN 1 OU")
    print("="*70)
    
    # =========================================================================
    # STEP 1: LOAD DATA
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 1: DATA LOADING")
    print("="*70)
    
    train_df, test_df = load_data()
    
    # =========================================================================
    # STEP 2: CLEAN DATA
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 2: DATA CLEANING")
    print("="*70)
    
    train_df = clean_data(train_df)
    
    # =========================================================================
    # STEP 3: FEATURE ENGINEERING
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 3: FEATURE ENGINEERING")
    print("="*70)
    
    train_features = extract_all_features(train_df)
    test_features = extract_all_features(test_df)
    
    # Add features to DataFrames
    for col in train_features.columns:
        train_df[col] = train_features[col].astype(float)
        test_df[col] = test_features[col].astype(float)
    
    # =========================================================================
    # STEP 4: PREPARE DATASET
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 4: DATASET PREPARATION")
    print("="*70)
    
    base_feature_cols = list(train_features.columns)
    y_train = train_df[COL_TARGET].astype(int)
    X_train = train_df[base_feature_cols].fillna(-1.0)
    X_test = test_df[base_feature_cols].fillna(-1.0)
    
    print(f"\n✓ Final dataset:")
    print(f"  - Total features: {len(base_feature_cols)}")
    print(f"  - X_train shape: {X_train.shape}")
    print(f"  - X_test shape: {X_test.shape}")
    print(f"  - Class balance: {y_train.mean():.3f}")
    
    # =========================================================================
    # STEP 5: FEATURE SELECTION
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 5: FEATURE SELECTION")
    print("="*70)
    
    selected_features = perform_feature_selection(X_train, y_train, base_feature_cols)
    
    # Prepare feature sets for each model
    log_features = selected_features['log_features']
    xgb_features = selected_features['xgb_features']
    
    X_train_log = train_df[log_features].fillna(-1.0)
    X_test_log = test_df[log_features].fillna(-1.0)
    
    X_train_xgb = train_df[xgb_features].fillna(-1.0)
    X_test_xgb = test_df[xgb_features].fillna(-1.0)
    
    print(f"\n✓ Feature sets per model:")
    print(f"  - LogReg: {len(log_features)} features")
    print(f"  - XGBoost: {len(xgb_features)} features")
    
    # =========================================================================
    # STEP 6: HYPERPARAMETER TUNING
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 6: HYPERPARAMETER TUNING")
    print("="*70)
    
    tuning_results = perform_hyperparameter_tuning(X_train_log, y_train, X_train_xgb, cv_folds=10)
    
    # Extract best parameters
    log_best_params = tuning_results['log_params']
    xgb_best_params = tuning_results['xgb_params']
    
    # =========================================================================
    # STEP 7: CROSS-VALIDATION WITH OPTIMIZED MODELS
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 7: CROSS-VALIDATION WITH OPTIMIZED MODELS")
    print("="*70)
    
    cv_results = cross_validate_models(
        X_train_log, y_train, X_train_xgb, 
        X_test_log, X_test_xgb,
        log_best_params, xgb_best_params, 
        n_folds=10
    )
    
    oof_log = cv_results['oof_log']
    oof_xgb = cv_results['oof_xgb']
    test_pred_log = cv_results['test_pred_log']
    test_pred_xgb = cv_results['test_pred_xgb']
    
    # =========================================================================
    # STEP 8: ENSEMBLE WEIGHT OPTIMIZATION
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 8: ENSEMBLE WEIGHT OPTIMIZATION")
    print("="*70)
    
    best_alpha, best_score, blend_results = optimize_ensemble_weights(
        oof_log, oof_xgb, y_train
    )
    
    # Create weighted blend predictions
    test_pred_blend = best_alpha * test_pred_log + (1 - best_alpha) * test_pred_xgb
    
    # =========================================================================
    # STEP 9: STACKING ENSEMBLE
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 9: STACKING ENSEMBLE")
    print("="*70)
    
    meta_model, meta_pred_train, test_pred_stack = create_stacking_ensemble(
        oof_log, oof_xgb, y_train, test_pred_log, test_pred_xgb
    )
    
    # Compare ensemble methods
    best_method = compare_ensemble_methods(
        oof_log, oof_xgb, y_train, best_alpha, meta_pred_train
    )
    
    # Choose final predictions based on best method
    if best_method == 'stacking':
        final_test_pred = test_pred_stack
        print("\n✓ Using STACKING for final submission")
    else:
        final_test_pred = test_pred_blend
        print("\n✓ Using WEIGHTED AVERAGE for final submission")
    
    # =========================================================================
    # STEP 10: LEARNING CURVES & OVERFITTING ANALYSIS
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 10: LEARNING CURVES & OVERFITTING ANALYSIS")
    print("="*70)
    
    overfitting_analysis = analyze_overfitting(
        X_train_log, y_train, X_train_xgb, 
        log_best_params, xgb_best_params
    )
    
    # =========================================================================
    # STEP 11: TRAIN FINAL MODELS FOR FEATURE IMPORTANCE
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 11: TRAINING FINAL MODELS")
    print("="*70)
    
    print("\n✓ Training final Logistic Regression...")
    clean_log_params = {k.replace('clf__', ''): v for k, v in log_best_params.items()}
    log_final = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(**clean_log_params, random_state=SEED, 
                                  max_iter=5000, solver='saga'))
    ])
    log_final.fit(X_train_log, y_train)
    
    print("✓ Training final XGBoost...")
    xgb_final = XGBClassifier(**xgb_best_params, random_state=SEED, eval_metric='logloss')
    xgb_final.fit(X_train_xgb, y_train)
    
    # =========================================================================
    # STEP 12: FEATURE IMPORTANCE ANALYSIS
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 12: FEATURE IMPORTANCE ANALYSIS")
    print("="*70)
    
    importance_results = perform_feature_importance_analysis(
        log_final, xgb_final, log_features, xgb_features, top_n=20
    )
    
    # =========================================================================
    # STEP 13: CREATE SUBMISSIONS
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 13: GENERATING SUBMISSIONS")
    print("="*70)
    
    create_submissions(
        test_df, test_pred_log, test_pred_xgb, 
        test_pred_blend, test_pred_stack
    )
    
    # Create main submission file (best method)
    submission = pd.DataFrame({
        COL_ID: test_df[COL_ID],
        COL_TARGET: (final_test_pred >= 0.5).astype(int)
    })
    submission.to_csv('submission.csv', index=False)
    print(f"\n✓ submission.csv (FINAL - {best_method.upper()})")
    print(f"  - Win rate: {submission[COL_TARGET].mean():.3f}")
    
    # =========================================================================
    # STEP 14: SUMMARY REPORT
    # =========================================================================
    print("\n" + "="*70)
    print("PIPELINE SUMMARY")
    print("="*70)
    
    print(f"\n✓ Data:")
    print(f"  - Training battles: {len(train_df)}")
    print(f"  - Test battles: {len(test_df)}")
    print(f"  - Total features engineered: {len(base_feature_cols)}")
    
    print(f"\n✓ Feature Selection:")
    print(f"  - LogReg features: {len(log_features)}")
    print(f"  - XGBoost features: {len(xgb_features)}")
    print(f"  - Common features: {len(set(log_features) & set(xgb_features))}")
    
    print(f"\n✓ Model Performance (Out-of-Fold):")
    acc_log_oof = accuracy_score(y_train, (oof_log >= 0.5).astype(int))
    acc_xgb_oof = accuracy_score(y_train, (oof_xgb >= 0.5).astype(int))
    acc_blend_oof = accuracy_score(y_train, (best_alpha * oof_log + (1-best_alpha) * oof_xgb >= 0.5).astype(int))
    acc_stack_oof = accuracy_score(y_train, (meta_pred_train >= 0.5).astype(int))
    
    print(f"  - LogReg: {acc_log_oof:.4f}")
    print(f"  - XGBoost: {acc_xgb_oof:.4f}")
    print(f"  - Weighted Ensemble: {acc_blend_oof:.4f}")
    print(f"  - Stacking: {acc_stack_oof:.4f}")
    
    print(f"\n✓ Ensemble Configuration:")
    print(f"  - Best method: {best_method.upper()}")
    if best_method == 'weighted':
        print(f"  - LogReg weight: {best_alpha:.2f}")
        print(f"  - XGBoost weight: {1-best_alpha:.2f}")
    else:
        print(f"  - Meta-learner: Logistic Regression")
        print(f"  - LogReg coefficient: {meta_model.coef_[0][0]:+.4f}")
        print(f"  - XGBoost coefficient: {meta_model.coef_[0][1]:+.4f}")
    
    print(f"\n✓ Output Files Generated:")
    output_files = [
        "submission.csv (FINAL)",
        "submission_logistic.csv",
        "submission_xgboost.csv",
        "submission_weighted_ensemble.csv",
        "submission_stacking.csv",
        "high_correlation_report.csv",
        "logreg_gridsearch_results.csv",
        "xgboost_gridsearch_results.csv",
        "best_hyperparameters.txt",
        "cv_fold_results.csv",
        "ensemble_weight_search.csv",
        "overfitting_analysis.csv",
        "logreg_feature_importance.csv",
        "xgboost_feature_importance.csv",
        "feature_importance_comparison.csv",
        "logistic_feature_importance.png",
        "xgboost_feature_importance.png",
        "feature_importance_comparison.png",
        "learning_curve_logistic.png",
        "learning_curve_xgboost.png"
    ]
    
    for i, file in enumerate(output_files, 1):
        print(f"  {i:2d}. {file}")
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    return {
        'train_df': train_df,
        'test_df': test_df,
        'log_model': log_final,
        'xgb_model': xgb_final,
        'meta_model': meta_model,
        'log_features': log_features,
        'xgb_features': xgb_features,
        'best_method': best_method,
        'best_alpha': best_alpha,
        'submission': submission,
        'cv_results': cv_results,
        'importance_results': importance_results,
        'overfitting_analysis': overfitting_analysis
    }


if __name__ == "__main__":
    results = main()
