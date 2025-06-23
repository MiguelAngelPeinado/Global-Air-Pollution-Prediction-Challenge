# Global Air Pollution Prediction Challenge

**Miguel Angel Peinado**  
📈 Current Public Leaderboard Position: 29th (Score: 0.8702)  

---

## 🚀 Project Overview

This repository contains my end-to-end solution for the **Global Air Pollution Prediction Challenge** on Bitgrit, where the goal is to forecast future air‐pollution levels at anonymized monitoring stations worldwide. Predictive accuracy is measured by RMSE (and leaderboard score = exp(−RMSE/100)).

Key steps taken:
1. **Data loading & cleaning**  
   - Load `train.csv` (7 features + target) and `test.csv`.  
   - Handle missing latitude/longitude via mean imputation.  
2. **Feature engineering**  
   - Cyclical encoding for hour, day‐of‐year, month.  
   - Interaction term `weekday_x_month`.  
   - Seasonal flags (e.g. `is_january`) and geographic clustering (`geo_cluster`) via K-means.  
3. **Dimensionality reduction**  
   - PCA on combined spatio-temporal features to capture latent patterns.  
4. **Cluster-wise modeling**  
   - Identify 10 geographic clusters.  
   - For each cluster:  
     - Train both Random Forest & XGBoost on the log‐transformed target.  
     - Evaluate in‐cluster RMSE and select XGBoost as the stronger “all-terrain” regressor (mean RMSE_XGB ≈ 0.105).  
   - Blend RF+XGB predictions with inverse‐RMSE weights per cluster.  
5. **Validation**  
   - Rigorous cross-validation (K-fold & stratified by cluster).  
   - Global CV RMSE ≈ **0.504**, cluster-XGB mean RMSE ≈ **0.105**.  

---

## 📁 Repository Structure
├── Python/
│ ├── AirPollution_Prediction_Challenge_Advanced_Ensemble_Validation.ipynb
│ ├── kmeans_model.pkl
│ ├── pca_model.pkl
│ ├── scaler_pca.pkl
│ ├── optimized_model_cluster_3.pkl
│ ├── optimized_model_cluster_5.pkl
│ ├── optimized_model_cluster_8.pkl
│ ├── rf_model_cluster_.pkl
│ └── xgb_model_cluster_.pkl
├── Data/
│ ├── train.csv
│ └── test.csv
└── README.md


🎯 Results & Metrics
Global cross-validation RMSE (log-space): ~0.504

Mean XGBoost RMSE per cluster: 0.105

Public leaderboard score: 0.8702 (Position 29/172)

Per‐cluster highlights:

Cluster 6: RMSE_XGB = 0.0206

Cluster 3: RMSE_XGB = 0.0920

Cluster 1: RMSE_XGB = 0.1370

🔭 Next Steps
Further feature engineering: meteorological data, lag-features, spatial interpolation.

Deep learning: Temporal CNNs/LSTMs for capturing time dependencies.

Stacking & meta-learning: richer ensembles beyond simple blending.

Hyperparameter tuning: Bayesian optimization (e.g. Optuna) for XGBoost per‐cluster.

🤝 Contact
Feel free to open issues or pull-requests. Discussion and feedback are very welcome!
📬 miguel.ang.peinado@gmail.com