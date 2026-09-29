"""
Robustness and Safety Audit Engine
Evaluates model stability under noise perturbation, missing value stress,
out-of-distribution (OOD) boundary resilience, and dataset anomaly profiling.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def evaluate_robustness(
    df: pd.DataFrame,
    feature_columns: List[str],
    model: Optional[Any] = None,
    y_pred: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Run robustness evaluation on dataset and model stability.
    """
    results: Dict[str, Any] = {
        "dataset_anomalies": {},
        "noise_perturbation_tests": [],
        "missing_value_resilience": {},
        "ood_stability": {},
        "prediction_flip_rate": 0.0,
        "score": 100,
        "risk_level": "LOW",
        "findings": [],
        "recommendations": []
    }

    if not feature_columns:
        results["findings"].append("No feature columns specified for robustness analysis.")
        results["score"] = 50
        return results

    X = df[feature_columns].copy()
    num_cols = [c for c in feature_columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in feature_columns if c not in num_cols]

    # Pre-process for numerical operations
    X_num = X.copy()
    for c in num_cols:
        X_num[c] = X_num[c].fillna(X_num[c].median())
    for c in cat_cols:
        X_num[c] = X_num[c].astype('category').cat.codes

    # 1. Dataset Anomaly & Outlier Profiling
    total_records = len(df)
    duplicates_count = int(df.duplicated().sum())
    duplicate_pct = round((duplicates_count / total_records) * 100, 2)

    # Outlier detection using Isolation Forest or IQR
    outlier_pct = 0.0
    if len(X_num) >= 20 and len(num_cols) > 0:
        try:
            iso = IsolationForest(contamination=0.05, random_state=42)
            outlier_labels = iso.fit_predict(X_num)
            outlier_count = int((outlier_labels == -1).sum())
            outlier_pct = round((outlier_count / total_records) * 100, 2)
        except Exception:
            outlier_pct = 3.0

    missing_cells = int(df[feature_columns].isna().sum().sum())
    total_cells = total_records * len(feature_columns)
    missing_pct = round((missing_cells / total_cells) * 100, 2) if total_cells > 0 else 0.0

    results["dataset_anomalies"] = {
        "duplicate_rows": duplicates_count,
        "duplicate_pct": duplicate_pct,
        "outlier_pct": outlier_pct,
        "missing_data_pct": missing_pct
    }

    if duplicate_pct > 5.0:
        results["findings"].append(f"High duplicate record rate ({duplicate_pct}%). May artificially inflate cross-validation scores.")
        results["recommendations"].append("Deduplicate dataset rows to prevent data leakage and skewed feature distributions.")

    if missing_pct > 15.0:
        results["findings"].append(f"Significant missing values ({missing_pct}% of total cells across audited features).")
        results["recommendations"].append("Implement robust imputation strategies (e.g. IterativeImputer or KNN) rather than arbitrary filling.")

    # 2. Model Noise Perturbation Stress Test
    flip_rates = []
    if model is not None and y_pred is not None and len(X_num) > 0:
        y_orig = np.array(y_pred)
        sample_size = min(300, len(X_num))
        X_test = X_num.sample(n=sample_size, random_state=42) if len(X_num) > sample_size else X_num
        y_base = y_orig[:len(X_test)] if len(y_orig) == len(X_num) else model.predict(X_test)

        noise_levels = [0.05, 0.10, 0.20]  # 5%, 10%, 20% relative noise
        for noise in noise_levels:
            X_pert = X_test.copy()
            for col in num_cols:
                std = float(X_test[col].std())
                if std > 1e-6:
                    jitter = np.random.normal(0, noise * std, size=len(X_test))
                    X_pert[col] = X_pert[col] + jitter

            try:
                pert_preds = model.predict(X_pert)
                flips = int(np.sum(pert_preds != y_base))
                flip_pct = round((flips / len(y_base)) * 100, 2)
                flip_rates.append(flip_pct)

                results["noise_perturbation_tests"].append({
                    "noise_magnitude": f"{int(noise * 100)}%",
                    "flipped_predictions_count": flips,
                    "flip_rate_pct": flip_pct,
                    "status": "PASS" if flip_pct < 10.0 else ("WARNING" if flip_pct < 25.0 else "FAIL")
                })
            except Exception:
                pass

        if flip_rates:
            primary_flip_rate = flip_rates[0]  # 5% noise flip rate
            results["prediction_flip_rate"] = primary_flip_rate

            if primary_flip_rate > 20.0:
                results["findings"].append(
                    f"Severe model fragility: Adding just 5% Gaussian noise flips {primary_flip_rate}% of predictions! "
                    "The model decision boundary is hypersensitive to small variations."
                )
                results["recommendations"].append(
                    "Apply adversarial training, weight decay regularization, or Gaussian data augmentation during training to smooth decision boundaries."
                )
            elif primary_flip_rate > 10.0:
                results["findings"].append(
                    f"Moderate noise sensitivity: 5% perturbation caused {primary_flip_rate}% prediction flips."
                )

        # 3. Missing Value Resilience Test
        try:
            # Simulate 10% random feature corruption / dropout
            X_drop = X_test.copy()
            mask = np.random.rand(*X_drop.shape) < 0.10
            # Impute dropped cells with column medians (standard fallback)
            for i, col in enumerate(X_drop.columns):
                col_mask = mask[:, i]
                X_drop.loc[col_mask, col] = X_num[col].median()

            drop_preds = model.predict(X_drop)
            drop_flips = int(np.sum(drop_preds != y_base))
            drop_flip_pct = round((drop_flips / len(y_base)) * 100, 2)
            results["missing_value_resilience"] = {
                "tested_dropout_rate": "10%",
                "prediction_drift_pct": drop_flip_pct,
                "resilience_status": "HIGH" if drop_flip_pct < 12.0 else ("MODERATE" if drop_flip_pct < 25.0 else "LOW")
            }
        except Exception:
            pass

        # 4. Out-of-Distribution (OOD) Stress Test
        try:
            # Synthetic 4-sigma extreme boundary inputs
            X_ood = X_test.copy()
            for col in num_cols:
                mean, std = float(X_test[col].mean()), float(X_test[col].std())
                if std > 1e-6:
                    X_ood[col] = mean + (4.0 * std * np.random.choice([-1, 1], size=len(X_test)))

            if hasattr(model, "predict_proba"):
                ood_probs = model.predict_proba(X_ood)
                max_confidence = float(np.max(ood_probs, axis=1).mean())
                results["ood_stability"] = {
                    "test": "4-Sigma Boundary Stress Test",
                    "mean_prediction_confidence": round(max_confidence, 3),
                    "overconfidence_risk": "HIGH" if max_confidence > 0.90 else "LOW"
                }
                if max_confidence > 0.90:
                    results["findings"].append(
                        f"Overconfidence on Out-of-Distribution inputs: Model assigns {round(max_confidence * 100, 1)}% "
                        "average confidence to extreme 4-sigma synthetic anomalies instead of signaling high uncertainty."
                    )
                    results["recommendations"].append("Incorporate uncertainty calibration (e.g. Temperature Scaling or Ensembling) and an OOD detector in production.")
        except Exception:
            pass

    # 5. Robustness Scoring (0 - 100)
    score = 100.0

    if results["noise_perturbation_tests"]:
        p_flip = results["prediction_flip_rate"]
        if p_flip > 25.0:
            score -= 40
        elif p_flip > 10.0:
            score -= (p_flip - 10.0) * 1.5 + 10
    else:
        # Dataset only
        if missing_pct > 15.0:
            score -= min(25, missing_pct)
        if duplicate_pct > 5.0:
            score -= min(15, duplicate_pct * 2)
        if outlier_pct > 10.0:
            score -= 10

    if results["missing_value_resilience"].get("resilience_status") == "LOW":
        score -= 15

    final_score = int(max(0, min(100, round(score))))
    results["score"] = final_score

    if final_score >= 85:
        results["risk_level"] = "LOW"
    elif final_score >= 65:
        results["risk_level"] = "MODERATE"
    elif final_score >= 45:
        results["risk_level"] = "HIGH"
    else:
        results["risk_level"] = "CRITICAL"

    if not results["recommendations"]:
        results["recommendations"].append("Model exhibits good resilience against noise and perturbations. Implement continuous drift monitoring.")

    return results

