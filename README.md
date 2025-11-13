# fds-pokemon-Log-karp-Crew
Prediction project

# Description
This project focuses on a binary classification task: predicting the winner (player_won) of competitive Pokémon battles.
he objective is to develop a robust machine learning model, capable of achieving high accuracy in predicting the battle outcome based on intricate game state and strategic features.

# Dataset
The dataset is provided in JSON Line files (train.jsonl, test.jsonl).
Training Set: 10,000 battle entries.
Test Set: 5,000 battle entries.
Features: The raw data centers on the battle_timeline, a chronological record containing details such as the Pokémon's name, current HP percentage (hp_pct), applied status conditions (status), and the moves executed by each player for every turn. The target variable is player_won.

# Workflow
1) Feature Engineering
A total of 184 domain-specific features were engineered from the raw timeline data. These features include:
--->Strategic Metrics: Role composition of surviving Pokémon at different turns (e.g., walls, special/physical breakers).
--->Battle Dynamics: for example status condition severity, turns disabled by status, damage tracking (total damage, variance, KO efficiency), and HP distribution (weighted averages, dispersion).

2) Feature Selection
A pruning process, involving correlation removal and GridSearch-optimized SelectFromModel (using Logistic Regression's coefficients and XGBoost's feature importance), reduced the final feature set.

3)Models
--->Base Learners: A Logistic Regression model (optimized via GridSearch for L1/L2/ElasticNet penalties) and an XGBoost Classifier (optimized via GridSearch for hyperparameters like n_estimators and max_depth).

Stacked Ensemble (Final)	0.8493

--->Final Predictor: A Stacking Classifier was implemented, using the optimized Logistic Regression model as the meta-learner to combine the predictions of the two base models.

--->Validation: Hyperparameter tuning and model evaluation employed a 5-fold Stratified K-Fold cross-validation strategy. The primary scoring metric used throughout the tuning process was Accuracy.

# Results


