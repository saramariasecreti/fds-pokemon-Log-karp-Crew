"""
Pokemon Battle Predictor - Feature Importance Analysis
======================================================
Comprehensive feature importance analysis and visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.libraries import *
from src.constants import *


def analyze_logreg_importance(model, feature_names, top_n=20):
    """
    Analyze and visualize Logistic Regression feature importance.
    
    Args:
        model: Trained LogisticRegression model (or pipeline with 'clf' step)
        feature_names: List of feature names
        top_n: Number of top features to display (default: 20)
        
    Returns:
        DataFrame: Feature importance dataframe
    """
    print("\n" + "="*70)
    print("LOGISTIC REGRESSION - FEATURE IMPORTANCE")
    print("="*70)
    
    # Extract coefficients
    if hasattr(model, 'named_steps'):
        # It's a pipeline
        coefficients = model.named_steps['clf'].coef_[0]
    else:
        coefficients = model.coef_[0]
    
    # Create importance dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients,
        'abs_coefficient': np.abs(coefficients)
    }).sort_values('abs_coefficient', ascending=False)
    
    print(f"\n✓ Top {min(15, len(importance_df))} Features:")
    print(importance_df.head(15).to_string(index=False))
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Top features by absolute coefficient
    top_features = importance_df.head(top_n)
    colors = ['#2ecc71' if c > 0 else '#e74c3c' for c in top_features['coefficient']]
    
    ax1.barh(range(top_n), top_features['abs_coefficient'], color=colors, alpha=0.7)
    ax1.set_yticks(range(top_n))
    ax1.set_yticklabels(top_features['feature'], fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlabel('Absolute Coefficient Value', fontsize=11, fontweight='bold')
    ax1.set_title(f'Top {top_n} Features - Logistic Regression\n(Green=Positive, Red=Negative)', 
                  fontsize=12, fontweight='bold', pad=15)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    ax1.axvline(x=0, color='black', linewidth=0.8)
    
    # Add value labels
    for i, (_, row) in enumerate(top_features.iterrows()):
        ax1.text(row['abs_coefficient'] + 0.01, i, f"{row['coefficient']:.3f}", 
                 va='center', fontsize=8, fontweight='bold')
    
    # Plot 2: Coefficient distribution
    ax2.hist(importance_df['coefficient'], bins=30, color='#3498db', 
             alpha=0.7, edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero')
    ax2.set_xlabel('Coefficient Value', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax2.set_title('Distribution of Feature Coefficients\nLogistic Regression', 
                  fontsize=12, fontweight='bold', pad=15)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3, linestyle='--')
    
    # Add statistics box
    stats_text = f"Mean: {importance_df['coefficient'].mean():.4f}\n"
    stats_text += f"Std: {importance_df['coefficient'].std():.4f}\n"
    stats_text += f"Positive: {(importance_df['coefficient'] > 0).sum()}\n"
    stats_text += f"Negative: {(importance_df['coefficient'] < 0).sum()}"
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
             fontsize=9, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('logistic_feature_importance.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved: logistic_feature_importance.png")
    
    # Save to CSV
    importance_df.to_csv('logreg_feature_importance.csv', index=False)
    print("✓ Data saved: logreg_feature_importance.csv")
    
    return importance_df


def analyze_xgboost_importance(model, feature_names, top_n=20):
    """
    Analyze and visualize XGBoost feature importance.
    
    Args:
        model: Trained XGBoost model
        feature_names: List of feature names
        top_n: Number of top features to display (default: 20)
        
    Returns:
        DataFrame: Feature importance dataframe
    """
    print("\n" + "="*70)
    print("XGBOOST - FEATURE IMPORTANCE")
    print("="*70)
    
    # Extract feature importance (gain-based)
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n✓ Top {min(15, len(importance_df))} Features:")
    print(importance_df.head(15).to_string(index=False))
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Top features by importance
    top_features = importance_df.head(top_n)
    colors_xgb = plt.cm.viridis(np.linspace(0.3, 0.9, top_n))
    
    ax1.barh(range(top_n), top_features['importance'], color=colors_xgb, alpha=0.8)
    ax1.set_yticks(range(top_n))
    ax1.set_yticklabels(top_features['feature'], fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlabel('Importance (Gain)', fontsize=11, fontweight='bold')
    ax1.set_title(f'Top {top_n} Features - XGBoost\n(Gain-based Importance)', 
                  fontsize=12, fontweight='bold', pad=15)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add value labels
    for i, (_, row) in enumerate(top_features.iterrows()):
        ax1.text(row['importance'] + 0.001, i, f"{row['importance']:.4f}", 
                 va='center', fontsize=8, fontweight='bold')
    
    # Plot 2: Importance distribution
    ax2.hist(importance_df['importance'], bins=25, color='#9b59b6', 
             alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Importance Value', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax2.set_title('Distribution of Feature Importance\nXGBoost (Gain)', 
                  fontsize=12, fontweight='bold', pad=15)
    ax2.grid(alpha=0.3, linestyle='--')
    
    # Add statistics box
    stats_text = f"Mean: {importance_df['importance'].mean():.4f}\n"
    stats_text += f"Std: {importance_df['importance'].std():.4f}\n"
    stats_text += f"Max: {importance_df['importance'].max():.4f}\n"
    stats_text += f"Non-zero: {(importance_df['importance'] > 0).sum()}"
    ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes, 
             fontsize=9, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('xgboost_feature_importance.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved: xgboost_feature_importance.png")
    
    # Save to CSV
    importance_df.to_csv('xgboost_feature_importance.csv', index=False)
    print("✓ Data saved: xgboost_feature_importance.csv")
    
    return importance_df


def compare_feature_importance(log_importance_df, xgb_importance_df, log_features, xgb_features):
    """
    Compare feature importance between LogReg and XGBoost.
    
    Args:
        log_importance_df: LogReg importance DataFrame
        xgb_importance_df: XGBoost importance DataFrame
        log_features: List of LogReg features
        xgb_features: List of XGBoost features
    """
    print("\n" + "="*70)
    print("COMPARATIVE FEATURE IMPORTANCE ANALYSIS")
    print("="*70)
    
    # Find common features
    common_features = set(log_features) & set(xgb_features)
    
    if not common_features:
        print("\n✗ No common features between models")
        return
    
    print(f"\n✓ Found {len(common_features)} common features")
    
    # Create ranking dataframes
    log_rank = log_importance_df.reset_index(drop=True).reset_index()
    log_rank.columns = ['log_rank', 'feature', 'coefficient', 'abs_coefficient']
    
    xgb_rank = xgb_importance_df.reset_index(drop=True).reset_index()
    xgb_rank.columns = ['xgb_rank', 'feature', 'importance']
    
    # Merge on common features
    comparison_df = pd.merge(
        log_rank[log_rank['feature'].isin(common_features)][['feature', 'log_rank', 'abs_coefficient']],
        xgb_rank[xgb_rank['feature'].isin(common_features)][['feature', 'xgb_rank', 'importance']],
        on='feature'
    )
    
    comparison_df['rank_diff'] = abs(comparison_df['log_rank'] - comparison_df['xgb_rank'])
    comparison_df = comparison_df.sort_values('rank_diff')
    
    print(f"\n✓ Features with most similar rankings (Top 10):")
    print(comparison_df.head(10).to_string(index=False))
    
    print(f"\n✓ Features with most different rankings (Top 10):")
    print(comparison_df.tail(10).to_string(index=False))
    
    # Create comparison scatter plot
    fig, ax = plt.subplots(figsize=(10, 10))
    
    scatter = ax.scatter(comparison_df['log_rank'], 
                        comparison_df['xgb_rank'],
                        c=comparison_df['abs_coefficient'],
                        s=comparison_df['importance']*5000,
                        alpha=0.6,
                        cmap='coolwarm',
                        edgecolors='black',
                        linewidth=0.5)
    
    # Add diagonal line (perfect agreement)
    max_rank = max(comparison_df['log_rank'].max(), comparison_df['xgb_rank'].max())
    ax.plot([0, max_rank], [0, max_rank], 'k--', alpha=0.3, linewidth=2, label='Perfect Agreement')
    
    ax.set_xlabel('Logistic Regression Rank', fontsize=12, fontweight='bold')
    ax.set_ylabel('XGBoost Rank', fontsize=12, fontweight='bold')
    ax.set_title('Feature Importance Rankings Comparison\n(Size=XGB Importance, Color=LogReg Coefficient)', 
                 fontsize=13, fontweight='bold', pad=15)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, linestyle='--')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('LogReg |Coefficient|', fontsize=10, fontweight='bold')
    
    # Annotate top 5 most important features from each model
    top_log = comparison_df.nsmallest(5, 'log_rank')
    top_xgb = comparison_df.nsmallest(5, 'xgb_rank')
    
    annotated_features = set()
    for _, row in pd.concat([top_log, top_xgb]).iterrows():
        if row['feature'] not in annotated_features:
            ax.annotate(row['feature'], 
                       (row['log_rank'], row['xgb_rank']),
                       fontsize=7, alpha=0.7, 
                       xytext=(5, 5), textcoords='offset points')
            annotated_features.add(row['feature'])
    
    plt.tight_layout()
    plt.savefig('feature_importance_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Comparison plot saved: feature_importance_comparison.png")
    
    # Save comparison data
    comparison_df.to_csv('feature_importance_comparison.csv', index=False)
    print("✓ Comparison data saved: feature_importance_comparison.csv")
    
    # Calculate correlation between rankings
    rank_correlation = comparison_df[['log_rank', 'xgb_rank']].corr().iloc[0, 1]
    print(f"\n✓ Rank correlation (Pearson): {rank_correlation:.4f}")
    
    return comparison_df


def perform_feature_importance_analysis(log_model, xgb_model, log_features, xgb_features, top_n=20):
    """
    Complete feature importance analysis pipeline.
    
    Args:
        log_model: Trained LogReg model
        xgb_model: Trained XGBoost model
        log_features: List of LogReg feature names
        xgb_features: List of XGBoost feature names
        top_n: Number of top features to display
        
    Returns:
        dict: Dictionary with importance dataframes
    """
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*70)
    
    # Analyze LogReg importance
    log_importance_df = analyze_logreg_importance(log_model, log_features, top_n)
    
    # Analyze XGBoost importance
    xgb_importance_df = analyze_xgboost_importance(xgb_model, xgb_features, top_n)
    
    # Compare importances
    comparison_df = compare_feature_importance(
        log_importance_df, xgb_importance_df, log_features, xgb_features
    )
    
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE ANALYSIS COMPLETE")
    print("="*70)
    
    return {
        'log_importance': log_importance_df,
        'xgb_importance': xgb_importance_df,
        'comparison': comparison_df
    }
