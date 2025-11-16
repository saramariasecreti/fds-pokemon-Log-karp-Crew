"""
Pokemon Battle Predictor - Ensemble Optimization
================================================
Advanced cross-validation, ensemble blending, and stacking.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.libraries import *
from src.constants import *


def cross_validate_models(X_train_log, y_train, X_train_xgb, X_test_log, X_test_xgb, 
                          log_best_params, xgb_best_params, n_folds=10):
    """
    Perform k-fold cross-validation with optimized models.
    
    Args:
        X_train_log: Training features for LogReg
        y_train: Training labels
        X_train_xgb: Training features for XGBoost
        X_test_log: Test features for LogReg
        X_test_xgb: Test features for XGBoost
        log_best_params: Best LogReg parameters from GridSearch
        xgb_best_params: Best XGBoost parameters from GridSearch
        n_folds: Number of CV folds (default: 10)
        
    Returns:
        dict: Dictionary with OOF predictions and test predictions
    """
    print("\n" + "="*70)
    print("CROSS-VALIDATION WITH OPTIMIZED MODELS")
    print("="*70)
    print(f"\n✓ Using {n_folds}-fold stratified cross-validation")
    
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    
    # Initialize arrays
    oof_log = np.zeros(len(X_train_log))
    oof_xgb = np.zeros(len(X_train_xgb))
    test_pred_log = np.zeros(len(X_test_log))
    test_pred_xgb = np.zeros(len(X_test_xgb))
    
    fold_results = []
    
    # Cross-validation loop
    for fold, (tr_idx, va_idx) in enumerate(kfold.split(X_train_log, y_train), 1):
        print(f"\n✓ Processing Fold {fold}/{n_folds}...")
        
        # Logistic Regression
        X_tr_log, X_va_log = X_train_log.iloc[tr_idx], X_train_log.iloc[va_idx]
        y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]
        
        # Clean parameters (remove 'clf__' prefix if present)
        clean_log_params = {k.replace('clf__', ''): v for k, v in log_best_params.items()}
        
        log_fold = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(**clean_log_params, random_state=SEED, 
                                      max_iter=5000, solver='saga'))
        ])
        log_fold.fit(X_tr_log, y_tr)
        oof_log[va_idx] = log_fold.predict_proba(X_va_log)[:, 1]
        test_pred_log += log_fold.predict_proba(X_test_log)[:, 1] / n_folds
        
        # XGBoost
        X_tr_xgb, X_va_xgb = X_train_xgb.iloc[tr_idx], X_train_xgb.iloc[va_idx]
        
        xgb_fold = XGBClassifier(**xgb_best_params, random_state=SEED, eval_metric='logloss')
        xgb_fold.fit(X_tr_xgb, y_tr)
        oof_xgb[va_idx] = xgb_fold.predict_proba(X_va_xgb)[:, 1]
        test_pred_xgb += xgb_fold.predict_proba(X_test_xgb)[:, 1] / n_folds
        
        # Calculate metrics
        acc_log = accuracy_score(y_va, (oof_log[va_idx] >= 0.5).astype(int))
        acc_xgb = accuracy_score(y_va, (oof_xgb[va_idx] >= 0.5).astype(int))
        ll_log = log_loss(y_va, oof_log[va_idx])
        ll_xgb = log_loss(y_va, oof_xgb[va_idx])
        
        fold_results.append({
            'fold': fold,
            'log_acc': acc_log,
            'xgb_acc': acc_xgb,
            'log_ll': ll_log,
            'xgb_ll': ll_xgb
        })
        
        print(f"  - LogReg: ACC={acc_log:.4f}, LogLoss={ll_log:.4f}")
        print(f"  - XGBoost: ACC={acc_xgb:.4f}, LogLoss={ll_xgb:.4f}")
    
    # Overall OOF metrics
    print("\n" + "="*70)
    print("OVERALL OUT-OF-FOLD METRICS")
    print("="*70)
    
    acc_log = accuracy_score(y_train, (oof_log >= 0.5).astype(int))
    ll_log = log_loss(y_train, oof_log)
    print(f"\n✓ LogReg:  ACC={acc_log:.4f}, LogLoss={ll_log:.4f}")
    
    acc_xgb = accuracy_score(y_train, (oof_xgb >= 0.5).astype(int))
    ll_xgb = log_loss(y_train, oof_xgb)
    print(f"✓ XGBoost: ACC={acc_xgb:.4f}, LogLoss={ll_xgb:.4f}")
    
    # Save fold results
    fold_results_df = pd.DataFrame(fold_results)
    fold_results_df.to_csv('cv_fold_results.csv', index=False)
    print(f"\n✓ Fold results saved: cv_fold_results.csv")
    
    return {
        'oof_log': oof_log,
        'oof_xgb': oof_xgb,
        'test_pred_log': test_pred_log,
        'test_pred_xgb': test_pred_xgb,
        'fold_results': fold_results_df
    }


def optimize_ensemble_weights(oof_log, oof_xgb, y_train):
    """
    Search for optimal blending weights between models.
    
    Args:
        oof_log: Out-of-fold predictions from LogReg
        oof_xgb: Out-of-fold predictions from XGBoost
        y_train: Training labels
        
    Returns:
        tuple: (best_alpha, best_score, blend_results)
    """
    print("\n" + "="*70)
    print("ENSEMBLE WEIGHT OPTIMIZATION")
    print("="*70)
    
    print("\n✓ Searching for optimal blend weight...")
    
    best_alpha = 0.5
    best_score = 0.0
    blend_results = []
    
    for alpha in np.arange(0.2, 0.81, 0.02):
        oof_blend = alpha * oof_log + (1 - alpha) * oof_xgb
        preds = (oof_blend >= 0.5).astype(int)
        score = accuracy_score(y_train, preds)
        ll = log_loss(y_train, oof_blend)
        
        blend_results.append({
            'alpha': alpha,
            'log_weight': alpha,
            'xgb_weight': 1 - alpha,
            'accuracy': score,
            'log_loss': ll
        })
        
        if score > best_score:
            best_score = score
            best_alpha = alpha
            print(f"  ✓ New best: Alpha={alpha:.2f} (LogReg={alpha:.2f}, XGBoost={1-alpha:.2f}), "
                  f"Accuracy={score:.4f}, LogLoss={ll:.4f}")
    
    # Final blend with best weights
    oof_blend = best_alpha * oof_log + (1 - best_alpha) * oof_xgb
    acc_blend = accuracy_score(y_train, (oof_blend >= 0.5).astype(int))
    ll_blend = log_loss(y_train, oof_blend)
    
    print(f"\n✓ FINAL ENSEMBLE WEIGHTS:")
    print(f"  - LogReg: {best_alpha:.2f}")
    print(f"  - XGBoost: {1-best_alpha:.2f}")
    print(f"  - OOF Accuracy: {acc_blend:.4f}")
    print(f"  - OOF LogLoss: {ll_blend:.4f}")
    
    # Save blend results
    blend_results_df = pd.DataFrame(blend_results)
    blend_results_df.to_csv('ensemble_weight_search.csv', index=False)
    print(f"\n✓ Weight search results saved: ensemble_weight_search.csv")
    
    return best_alpha, best_score, blend_results_df


def create_stacking_ensemble(oof_log, oof_xgb, y_train, test_pred_log, test_pred_xgb):
    """
    Create stacking ensemble with meta-learner.
    
    Args:
        oof_log: Out-of-fold predictions from LogReg
        oof_xgb: Out-of-fold predictions from XGBoost
        y_train: Training labels
        test_pred_log: Test predictions from LogReg
        test_pred_xgb: Test predictions from XGBoost
        
    Returns:
        tuple: (meta_model, meta_pred_train, test_pred_stack)
    """
    print("\n" + "="*70)
    print("STACKING ENSEMBLE (META-LEARNING)")
    print("="*70)
    
    # Step 1: Create meta-features
    X_meta_train = np.column_stack([oof_log, oof_xgb])
    
    print(f"\n✓ Meta-features shape: {X_meta_train.shape}")
    print(f"  - oof_log: mean={oof_log.mean():.3f}, std={oof_log.std():.3f}")
    print(f"  - oof_xgb: mean={oof_xgb.mean():.3f}, std={oof_xgb.std():.3f}")
    
    # Step 2: Train meta-model on OOF predictions
    meta_model = LogisticRegression(C=1.0, random_state=SEED, max_iter=1000)
    meta_model.fit(X_meta_train, y_train)
    
    print(f"\n✓ Meta-model learned weights:")
    print(f"  - LogReg coefficient: {meta_model.coef_[0][0]:+.4f}")
    print(f"  - XGBoost coefficient: {meta_model.coef_[0][1]:+.4f}")
    print(f"  - Intercept: {meta_model.intercept_[0]:+.4f}")
    
    # Step 3: Meta predictions on training data
    meta_pred_train = meta_model.predict_proba(X_meta_train)[:, 1]
    acc_meta = accuracy_score(y_train, (meta_pred_train >= 0.5).astype(int))
    ll_meta = log_loss(y_train, meta_pred_train)
    
    print(f"\n✓ STACKING TRAIN PERFORMANCE:")
    print(f"  - Accuracy: {acc_meta:.4f}")
    print(f"  - LogLoss: {ll_meta:.4f}")
    
    # Step 4: Test predictions
    X_meta_test = np.column_stack([test_pred_log, test_pred_xgb])
    test_pred_stack = meta_model.predict_proba(X_meta_test)[:, 1]
    
    return meta_model, meta_pred_train, test_pred_stack


def compare_ensemble_methods(oof_log, oof_xgb, y_train, best_alpha, meta_pred_train):
    """
    Compare weighted average vs stacking ensemble.
    
    Args:
        oof_log: Out-of-fold predictions from LogReg
        oof_xgb: Out-of-fold predictions from XGBoost
        y_train: Training labels
        best_alpha: Best alpha from weight optimization
        meta_pred_train: Meta-model predictions
        
    Returns:
        str: Best ensemble method ('weighted' or 'stacking')
    """
    print("\n" + "="*70)
    print("ENSEMBLE METHOD COMPARISON")
    print("="*70)
    
    # Weighted average
    oof_blend = best_alpha * oof_log + (1 - best_alpha) * oof_xgb
    acc_blend = accuracy_score(y_train, (oof_blend >= 0.5).astype(int))
    ll_blend = log_loss(y_train, oof_blend)
    
    # Stacking
    acc_stack = accuracy_score(y_train, (meta_pred_train >= 0.5).astype(int))
    ll_stack = log_loss(y_train, meta_pred_train)
    
    print(f"\n✓ Weighted Average:")
    print(f"  - Accuracy: {acc_blend:.4f}")
    print(f"  - LogLoss: {ll_blend:.4f}")
    
    print(f"\n✓ Stacking:")
    print(f"  - Accuracy: {acc_stack:.4f}")
    print(f"  - LogLoss: {ll_stack:.4f}")
    
    print(f"\n✓ Improvement:")
    print(f"  - Accuracy: {acc_stack - acc_blend:+.4f}")
    print(f"  - LogLoss: {ll_blend - ll_stack:+.4f} (lower is better)")
    
    if acc_stack > acc_blend:
        print(f"\n✓ WINNER: Stacking ensemble performs better!")
        best_method = 'stacking'
    else:
        print(f"\n✓ WINNER: Weighted average performs better!")
        best_method = 'weighted'
    
    return best_method


def plot_learning_curves(estimator, X, y, title, cv=5):
    """
    Plot learning curves to detect overfitting.
    
    Args:
        estimator: Model to evaluate
        X: Features
        y: Labels
        title: Plot title
        cv: Number of CV folds (default: 5)
        
    Returns:
        tuple: (train_acc, val_acc, gap)
    """
    print(f"\n✓ Generating learning curve for {title}...")
    
    train_sizes = np.linspace(0.1, 1.0, 10)
    
    train_sizes_abs, train_scores, val_scores = learning_curve(
        estimator, X, y,
        train_sizes=train_sizes,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        random_state=SEED
    )
    
    # Calculate mean and std
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes_abs, train_mean, 'o-', color='r', label='Training score', linewidth=2)
    plt.plot(train_sizes_abs, val_mean, 'o-', color='g', label='Validation score', linewidth=2)
    
    # Shade std deviation
    plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, 
                     alpha=0.15, color='r')
    plt.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, 
                     alpha=0.15, color='g')
    
    plt.xlabel('Training Examples', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
    plt.title(f'Learning Curves - {title}', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add gap annotation
    final_gap = train_mean[-1] - val_mean[-1]
    plt.text(0.02, 0.98, f'Final Gap: {final_gap:.4f}', 
             transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', 
             facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    
    # Gap analysis
    print(f"  - Final Training Accuracy: {train_mean[-1]:.4f}")
    print(f"  - Final Validation Accuracy: {val_mean[-1]:.4f}")
    print(f"  - Train-Val Gap: {final_gap:.4f}")
    
    if final_gap > 0.05:
        print(f"  ⚠ WARNING: Possible overfitting detected!")
    elif final_gap < 0.01:
        print(f"  ✓ Good generalization")
    else:
        print(f"  ✓ Acceptable gap")
    
    return train_mean[-1], val_mean[-1], final_gap


def analyze_overfitting(X_train_log, y_train, X_train_xgb, log_best_params, xgb_best_params):
    """
    Analyze overfitting using learning curves.
    
    Args:
        X_train_log: Training features for LogReg
        y_train: Training labels
        X_train_xgb: Training features for XGBoost
        log_best_params: Best LogReg parameters
        xgb_best_params: Best XGBoost parameters
        
    Returns:
        DataFrame: Overfitting analysis summary
    """
    print("\n" + "="*70)
    print("LEARNING CURVES - OVERFITTING ANALYSIS")
    print("="*70)
    
    # 1. Logistic Regression Learning Curve
    clean_log_params = {k.replace('clf__', ''): v for k, v in log_best_params.items()}
    log_estimator = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(**clean_log_params, random_state=SEED, 
                                  max_iter=5000, solver='saga'))
    ])
    
    log_train_acc, log_val_acc, log_gap = plot_learning_curves(
        log_estimator, X_train_log, y_train, "Logistic Regression", cv=5
    )
    plt.savefig('learning_curve_logistic.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. XGBoost Learning Curve
    xgb_estimator = XGBClassifier(**xgb_best_params, random_state=SEED, eval_metric='logloss')
    xgb_train_acc, xgb_val_acc, xgb_gap = plot_learning_curves(
        xgb_estimator, X_train_xgb, y_train, "XGBoost", cv=5
    )
    plt.savefig('learning_curve_xgboost.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Summary comparison
    print("\n" + "="*70)
    print("OVERFITTING SUMMARY")
    print("="*70)
    
    comparison = pd.DataFrame({
        'Model': ['Logistic Regression', 'XGBoost'],
        'Train_Acc': [log_train_acc, xgb_train_acc],
        'Val_Acc': [log_val_acc, xgb_val_acc],
        'Gap': [log_gap, xgb_gap],
        'Status': [
            'Overfitting' if log_gap > 0.05 else 'OK',
            'Overfitting' if xgb_gap > 0.05 else 'OK'
        ]
    })
    
    print("\n" + comparison.to_string(index=False))
    
    print("\n✓ Learning curves saved:")
    print("  - learning_curve_logistic.png")
    print("  - learning_curve_xgboost.png")
    
    comparison.to_csv('overfitting_analysis.csv', index=False)
    print("  - overfitting_analysis.csv")
    
    return comparison


def create_submissions(test_df, test_pred_log, test_pred_xgb, test_pred_blend, test_pred_stack):
    """
    Create multiple submission files.
    
    Args:
        test_df: Test DataFrame
        test_pred_log: LogReg predictions
        test_pred_xgb: XGBoost predictions
        test_pred_blend: Weighted blend predictions
        test_pred_stack: Stacking predictions
    """
    print("\n" + "="*70)
    print("CREATING SUBMISSIONS")
    print("="*70)
    
    from src.constants import COL_ID, COL_TARGET
    
    # LogReg submission
    submission_log = pd.DataFrame({
        COL_ID: test_df[COL_ID],
        COL_TARGET: (test_pred_log >= 0.5).astype(int)
    })
    submission_log.to_csv('submission_logistic.csv', index=False)
    print(f"\n✓ submission_logistic.csv")
    print(f"  - Win rate: {submission_log[COL_TARGET].mean():.3f}")
    
    # XGBoost submission
    submission_xgb = pd.DataFrame({
        COL_ID: test_df[COL_ID],
        COL_TARGET: (test_pred_xgb >= 0.5).astype(int)
    })
    submission_xgb.to_csv('submission_xgboost.csv', index=False)
    print(f"\n✓ submission_xgboost.csv")
    print(f"  - Win rate: {submission_xgb[COL_TARGET].mean():.3f}")
    
    # Weighted blend submission
    submission_blend = pd.DataFrame({
        COL_ID: test_df[COL_ID],
        COL_TARGET: (test_pred_blend >= 0.5).astype(int)
    })
    submission_blend.to_csv('submission_weighted_ensemble.csv', index=False)
    print(f"\n✓ submission_weighted_ensemble.csv")
    print(f"  - Win rate: {submission_blend[COL_TARGET].mean():.3f}")
    
    # Stacking submission
    submission_stack = pd.DataFrame({
        COL_ID: test_df[COL_ID],
        COL_TARGET: (test_pred_stack >= 0.5).astype(int)
    })
    submission_stack.to_csv('submission_stacking.csv', index=False)
    print(f"\n✓ submission_stacking.csv")
    print(f"  - Win rate: {submission_stack[COL_TARGET].mean():.3f}")
    
    print(f"\n✓ All submissions created!")
