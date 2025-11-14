# Pokemon Battle Predictor 

Machine learning models for predicting outcomes of competitive Pokemon battles using feature engineering and ensemble methods.

## 🎯 Project Overview

This project analyzes Pokemon battle timelines from the first turns to predict match outcomes with high accuracy. It extracts 150+ strategic features covering damage patterns, status warfare, role composition, move quality, and complex interactions.

### Key Features

- **Comprehensive Feature Engineering**: 150+ features across different categories
- **Domain-Driven Design**: Features based on competitive Pokemon knowledge
- **Advanced Selection**: GridSearch-optimized feature selection with correlation pruning
- **Ensemble Methods**: LogisticRegression + XGBoost ensemble
- **Chess-Like Position Evaluation**: Detects errors and blunders in gameplay

## 📊 Feature Categories

| Category | Count | Description |
|----------|-------|-------------|
| Status Features | Timing, targets, and momentum of status conditions |
| Endgame Features | KOs, survivors, active field effects at T30 |
| Role Features | Survivor role composition (walls, breakers, spreaders) |
| HP Distribution | Average, variance, dispersion of survivor HP |
| Move Quality | Position evaluation, errors, blunders detection |
| Static Features | Pre-battle team composition analysis |
| Momentum Features | HP momentum and swing detection |
| Critical Pokemon | Survival of high-value Pokemon |
| Early Game | Opening aggression and damage patterns |
| Boost Features | Stat boost accumulation tracking |
| Interactions | Multiplicative feature interactions |

## 🗂️ Repository Structure

```
pokemon-battle-predictor/
├── libraries.py                 # Centralized imports
├── constants.py                 # Domain constants and configurations
├── functions.py                 # Feature engineering functions (154+ features)
├── feature_selection.py         # Feature selection pipeline with GridSearch
├── hyperparameter_tuning.py     # GridSearch hyperparameter optimization
├── ensemble_optimization.py     # Advanced CV, ensemble blending, stacking
├── feature_importance.py        # Feature importance analysis & visualization
├── main.py                      # Main execution pipeline (14 steps)
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT License
├── README.md                    # This file
```

### Usage

The pipeline will:
1. Load train.jsonl and test.jsonl from the current directory
2. Clean and preprocess data
3. Extract features across categories
4. Perform feature selection with GridSearch optimization
5. Optimize hyperparameters for both models (10-fold CV)
6. Perform 10-fold cross-validation with optimized models
7. Optimize ensemble weights (weighted average)
8. Create stacking ensemble with meta-learner
9. Compare ensemble methods and select best
10. Analyze overfitting with learning curves
11. Train final models on full dataset
12. Perform comprehensive feature importance analysis
13. Generate multiple submission files
14. Create detailed summary report

## 🔬 Feature Engineering Details

### Status Features
Tracks status condition applications (Sleep, Freeze, Paralysis, Burn) with emphasis on:
- **High-value targets**: Tauros, Alakazam, Chansey, etc.
- **Temporal weighting**: Recent status more impactful
- **Momentum shifts**: Late-game status swing detection

### HP Distribution
Analyzes survivor HP quality beyond simple averages:
- Standard deviation and coefficient of variation
- Weighted averages using Pokemon tier values
- Effective HP adjusted for status penalties
- Dispersion flags for concentrated vs spread damage

### Move Quality (Position Evaluation)
Chess-inspired position scoring system:
- Calculates position scores considering Pokemon value, HP, status, roles, tempo
- Detects errors (score drop > 1.0) and blunders (drop > 2.0)
- Tracks temporal distribution of mistakes

### Interaction Features
Multiplicative interactions capturing non-linear relationships:
- Momentum × Material (survivors, KOs)
- Critical Pokemon × HP Quality
- Boost × Survivors (setup sweeper success)
- Damage × Control (status synergies)

## 📈 Model Performance
## Ensemble Methods

The pipeline implements **three ensemble strategies**:

1. **Individual Models**
   - LogisticRegression with L1/L2/ElasticNet regularization
   - XGBoost with optimized tree parameters

2. **Weighted Average Ensemble**
   - Optimizes blend weights via grid search (α from 0.2 to 0.8)

3. **Stacking Ensemble (Meta-Learning)**
   - Uses out-of-fold predictions as meta-features
   - Trains LogisticRegression meta-learner
   - Often achieves +0.5-1% improvement over weighted average

**The pipeline automatically selects the best performing method.**

### Model Configuration

**Logistic Regression:**
- Penalty: L2
- C: 0.1
- Solver: saga
- Features: ~50-70 (after selection)

**XGBoost:**
- n_estimators: 800
- max_depth: 3
- learning_rate: 0.03
- Features: ~60-80 (after selection)

## 🎓 Technical Highlights

### Feature Selection Pipeline

1. **Correlation Pruning**: Removes features with |corr| > 0.9, keeping more important one
2. **GridSearch Optimization**: Finds optimal thresholds for SelectFromModel
3. **Dual Selection**: Separate feature sets for LogReg and XGBoost
4. **Union Strategy**: Combines selected features from both methods







