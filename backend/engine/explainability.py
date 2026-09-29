"""
Explainability and Interpretability Audit Engine
Evaluates feature attributions, model complexity, surrogate fidelity, and interpretability score.
Uses SHAP (SHapley Additive exPlanations) and surrogate tree modeling.
"""

from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.inspection import permutation_importance

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


def evaluate_explainability(
    df: pd.DataFrame,
    feature_columns: List[str],
    model: Optional[Any] = None,
    y_pred: Optional[Union[np.ndarray, pd.Series, List]] = None,
    task_type: str = "classification"
) -> Dict[str, Any]:
    """
    Run comprehensive explainability evaluation on model and features.
    """
    results: Dict[str, Any] = {
        "model_type": "Unknown",
        "complexity_level": "UNKNOWN",
        "feature_attributions": [],
        "surrogate_fidelity": None,
        "feature_concentration": None,
        "score": 100,
        "risk_level": "LOW",
        "findings": [],
        "recommendations": []
    }

    if not feature_columns:
        results["findings"].append("No feature columns specified for explainability analysis.")
        results["score"] = 50
        return results

    X = df[feature_columns].copy()
    # Fill numeric NaNs with median, categoricals with mode for explainer stability
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].fillna(X[col].median())
        else:
            X[col] = X[col].astype('category').cat.codes

    # 1. Model Complexity & Family Analysis
    model_family = "Unknown / Dataset Only"
    inherent_interpretability = "MODERATE"
    base_complexity_score = 80

    if model is not None:
        model_name = model.__class__.__name__
        results["model_type"] = model_name

        if any(term in model_name.lower() for term in ["logistic", "linear", "ridge", "lasso"]):
            model_family = "Linear / White-Box Model"
            inherent_interpretability = "HIGH"
            base_complexity_score = 95
        elif any(term in model_name.lower() for term in ["decisiontree"]):
            model_family = "Single Decision Tree"
            inherent_interpretability = "HIGH"
            base_complexity_score = 90
        elif any(term in model_name.lower() for term in ["randomforest", "extratrees", "gradientboost", "histgradientboost", "adaboost", "lgbm", "xgb", "catboost"]):
            model_family = "Tree Ensemble (Random Forest / Gradient Boosted)"
            inherent_interpretability = "MODERATE"
            base_complexity_score = 75
        elif any(term in model_name.lower() for term in ["svc", "svm", "mlp", "neural", "sequential", "deep"]):
            model_family = "Black-Box Non-Linear / Neural Network"
            inherent_interpretability = "LOW"
            base_complexity_score = 55
        else:
            model_family = f"Custom Model ({model_name})"
            inherent_interpretability = "MODERATE"
            base_complexity_score = 70

        results["complexity_level"] = inherent_interpretability
        results["model_family"] = model_family
        results["findings"].append(f"Model identified as '{model_name}' ({model_family}) with {inherent_interpretability} inherent interpretability.")
    else:
        results["complexity_level"] = "HIGH"
        results["model_family"] = "Dataset Features Analysis"
        results["findings"].append("No pre-trained model provided. Performing dataset feature structure and correlation audit.")

    # 2. Global Feature Attributions (SHAP / Feature Importances)
    feature_importances: Dict[str, float] = {}
    attribution_method = "Permutation / Surrogate"

    if model is not None:
        # Try SHAP first
        shap_computed = False
        if SHAP_AVAILABLE:
            try:
                # Use a sample of up to 150 rows for speed and responsiveness
                sample_size = min(150, len(X))
                X_sample = X.sample(n=sample_size, random_state=42) if len(X) > sample_size else X

                if any(term in str(type(model)).lower() for term in ["tree", "forest", "boost"]):
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_sample)
                    if isinstance(shap_values, list):
                        # Multi-class or binary list
                        vals = np.abs(shap_values[1] if len(shap_values) > 1 else shap_values[0]).mean(axis=0)
                    elif len(np.shape(shap_values)) == 3:
                        vals = np.abs(shap_values[:, :, 1] if shap_values.shape[2] > 1 else shap_values[:, :, 0]).mean(axis=0)
                    else:
                        vals = np.abs(shap_values).mean(axis=0)
                    
                    for f, val in zip(feature_columns, vals):
                        feature_importances[f] = float(val)
                    attribution_method = "SHAP TreeExplainer"
                    shap_computed = True
                elif hasattr(model, "coef_"):
                    coefs = np.abs(model.coef_).flatten()
                    for f, c in zip(feature_columns, coefs[:len(feature_columns)]):
                        feature_importances[f] = float(c)
                    attribution_method = "Linear Coefficients (Direct Attribution)"
                    shap_computed = True
            except Exception:
                shap_computed = False

        # Fallback to model's feature_importances_ if available
        if not shap_computed and hasattr(model, "feature_importances_"):
            for f, imp in zip(feature_columns, model.feature_importances_):
                feature_importances[f] = float(imp)
            attribution_method = "Model Built-in Gini Importance"
            shap_computed = True

        # Fallback to permutation importance
        if not shap_computed and y_pred is not None:
            try:
                perm = permutation_importance(model, X, y_pred, n_repeats=3, random_state=42)
                for f, imp in zip(feature_columns, perm.importances_mean):
                    feature_importances[f] = float(max(0, imp))
                attribution_method = "Permutation Feature Importance"
            except Exception:
                pass
    else:
        # Dataset-level variance / correlation
        for col in feature_columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                feature_importances[col] = float(X[col].std())
            else:
                feature_importances[col] = float(X[col].nunique())
        attribution_method = "Feature Dispersion & Variance"

    # Normalize feature importances to percentages
    total_imp = sum(feature_importances.values())
    sorted_features = []
    if total_imp > 0:
        for f, imp in sorted(feature_importances.items(), key=lambda x: x[1], reverse=True):
            pct = round((imp / total_imp) * 100, 2)
            sorted_features.append({"feature": f, "importance": round(imp, 4), "percentage": pct})
    else:
        for f in feature_columns:
            sorted_features.append({"feature": f, "importance": 1.0, "percentage": round(100.0 / len(feature_columns), 2)})

    results["feature_attributions"] = sorted_features[:12]
    results["attribution_method"] = attribution_method

    # 3. Feature Concentration / Dominance (Gini Coefficient of Importances)
    if sorted_features:
        top1 = sorted_features[0]["percentage"]
        top3 = sum(f["percentage"] for f in sorted_features[:3])
        results["feature_concentration"] = {
            "top_feature": sorted_features[0]["feature"],
            "top_1_percentage": top1,
            "top_3_percentage": round(top3, 2)
        }

        if top1 > 60:
            results["findings"].append(
                f"Extreme feature concentration: Single feature '{sorted_features[0]['feature']}' accounts for "
                f"{top1}% of model decisions. Risk of single-point failure or hidden proxy bias."
            )
            results["recommendations"].append(
                f"Investigate '{sorted_features[0]['feature']}' to ensure it is not acting as an unvetted proxy or leaking target information."
            )
        elif top3 > 85:
            results["findings"].append(
                f"High feature concentration: Top 3 features drive {round(top3, 1)}% of all predictions."
            )

    # 4. White-Box Surrogate Explainer Fidelity
    if y_pred is not None and len(X) > 10:
        try:
            if task_type == "classification":
                surrogate = DecisionTreeClassifier(max_depth=3, random_state=42)
                surrogate.fit(X, y_pred)
                surr_preds = surrogate.predict(X)
                fidelity = float(accuracy_score(y_pred, surr_preds))
            else:
                surrogate = DecisionTreeRegressor(max_depth=3, random_state=42)
                surrogate.fit(X, y_pred)
                surr_preds = surrogate.predict(X)
                fidelity = float(r2_score(y_pred, surr_preds))

            results["surrogate_fidelity"] = {
                "metric": "Surrogate Decision Tree Accuracy" if task_type == "classification" else "Surrogate R2",
                "fidelity_score": round(fidelity, 3),
                "interpretable_depth": 3
            }

            if fidelity < 0.70:
                results["findings"].append(
                    f"Low white-box surrogate fidelity ({round(fidelity * 100, 1)}%): Complex non-linear interactions cannot be easily approximated by simple rules."
                )
                results["recommendations"].append("Provide instance-level SHAP force plots or counterfactual explanations to stakeholders.")
            else:
                results["findings"].append(
                    f"Good surrogate fidelity ({round(fidelity * 100, 1)}%): Model logic can be approximated by shallow decision rules for regulatory audits."
                )
        except Exception:
            pass

    # 5. Explainability Scoring (0 - 100)
    score = float(base_complexity_score)

    # Adjust based on surrogate fidelity
    if results["surrogate_fidelity"]:
        fid = results["surrogate_fidelity"]["fidelity_score"]
        if fid >= 0.85:
            score += 5
        elif fid < 0.65:
            score -= 15

    # Penalize extreme feature dominance
    if results["feature_concentration"]:
        if results["feature_concentration"]["top_1_percentage"] > 60:
            score -= 15
        elif results["feature_concentration"]["top_1_percentage"] > 45:
            score -= 5

    final_score = int(max(0, min(100, round(score))))
    results["score"] = final_score

    if final_score >= 80:
        results["risk_level"] = "LOW"
    elif final_score >= 65:
        results["risk_level"] = "MODERATE"
    elif final_score >= 45:
        results["risk_level"] = "HIGH"
    else:
        results["risk_level"] = "CRITICAL"

    if not results["recommendations"]:
        results["recommendations"].append("Model explainability is adequate. Provide SHAP feature importance charts in the governance documentation.")

    return results

