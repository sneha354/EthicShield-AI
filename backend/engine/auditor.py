"""
Master AI Ethics Auditor Orchestrator
Coordinates the 5 audit engines: Explainability, Transparency, Fairness, Privacy, and Robustness.
Supports Autonomous Zero-Configuration Auditing:
- Target column auto-detection
- Multi-attribute demographic discovery (Gender, Age, Race, etc.)
- Baseline privileged group statistical inference
- Feature schema alignment for scikit-learn serialized models
Computes overall AI Ethics Score, letter grade, risk assessment, and prioritized remediation roadmap.
"""

from typing import Dict, Any, List, Optional, Union
import datetime
import numpy as np
import pandas as pd

from .fairness import evaluate_fairness
from .explainability import evaluate_explainability
from .privacy import evaluate_privacy
from .robustness import evaluate_robustness
from .transparency import evaluate_transparency


DEFAULT_WEIGHTS = {
    "fairness": 0.25,
    "explainability": 0.20,
    "privacy": 0.20,
    "robustness": 0.20,
    "transparency": 0.15
}

TARGET_KEYWORDS = [
    "target", "label", "approved", "class", "outcome", "credit_risk",
    "readmitted_30d", "readmitted", "hired", "default", "churn", "status", "y"
]


def auto_detect_target_column(df: pd.DataFrame) -> Optional[str]:
    """Automatically detect the primary target/decision column in a dataset."""
    # 1. Exact or partial keyword match in column names
    for col in df.columns:
        c_lower = col.lower()
        if c_lower in TARGET_KEYWORDS or any(c_lower.endswith("_" + k) or c_lower.startswith(k + "_") for k in TARGET_KEYWORDS):
            return col
    # 2. Look for binary / low-cardinality outcome column (starting from the last column)
    for col in reversed(df.columns):
        if df[col].dropna().nunique() == 2:
            return col
    # 3. Fallback to the last column
    return df.columns[-1] if len(df.columns) > 0 else None


def get_letter_grade(score: float) -> str:
    if score >= 93:
        return "A+"
    elif score >= 85:
        return "A"
    elif score >= 75:
        return "B"
    elif score >= 65:
        return "C"
    elif score >= 50:
        return "D"
    else:
        return "F"


def get_risk_status(score: float) -> str:
    if score >= 80:
        return "LOW RISK (APPROVED)"
    elif score >= 65:
        return "MODERATE RISK (CONDITIONAL)"
    elif score >= 50:
        return "HIGH RISK (MITIGATION REQUIRED)"
    else:
        return "CRITICAL RISK (DEPLOYMENT BLOCKED)"


class AIEthicsAuditor:
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or DEFAULT_WEIGHTS

    def audit(
        self,
        df: pd.DataFrame,
        sensitive_column: Optional[str] = None,
        target_column: Optional[str] = None,
        feature_columns: Optional[List[str]] = None,
        model: Optional[Any] = None,
        y_true: Optional[Union[np.ndarray, pd.Series, List]] = None,
        y_pred: Optional[Union[np.ndarray, pd.Series, List]] = None,
        train_acc: Optional[float] = None,
        test_acc: Optional[float] = None,
        privileged_group: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute full multi-pillar AI ethics audit.
        Supports completely autonomous execution with zero required configuration.
        """
        audit_timestamp = datetime.datetime.now().isoformat()
        metadata = metadata or {}

        # 1. Target Column Auto-Detection
        if not target_column or str(target_column).lower() == "auto":
            target_column = auto_detect_target_column(df)

        # 2. Feature Columns Auto-Alignment
        if feature_columns is None:
            # If scikit-learn model has recorded feature names, match them directly
            if model is not None and hasattr(model, "feature_names_in_"):
                feature_columns = [c for c in model.feature_names_in_ if c in df.columns]

            if not feature_columns:
                excluded = []
                if target_column:
                    excluded.append(target_column)
                feature_columns = [c for c in df.columns if c not in excluded]

        # 3. Model Predictions Generation (if y_pred not provided)
        if model is not None and y_pred is None and feature_columns:
            try:
                X = df[feature_columns].copy()
                for c in X.columns:
                    if pd.api.types.is_numeric_dtype(X[c]):
                        X[c] = X[c].fillna(X[c].median())
                    else:
                        X[c] = X[c].astype('category').cat.codes
                y_pred = model.predict(X)
            except Exception:
                y_pred = None

        # 4. Ground Truth Labels
        if y_true is None and target_column and target_column in df.columns:
            y_true = df[target_column].values

        # 5. Pillar 1: Autonomous Multi-Demographic Fairness
        fairness_res = evaluate_fairness(
            df=df,
            sensitive_column=sensitive_column if sensitive_column and str(sensitive_column).lower() != "auto" else None,
            target_column=target_column,
            y_true=y_true,
            y_pred=y_pred,
            privileged_group=privileged_group if privileged_group and str(privileged_group).lower() != "auto" else None
        )

        # 6. Pillar 2: Explainability
        explainability_res = evaluate_explainability(
            df=df,
            feature_columns=feature_columns,
            model=model,
            y_pred=y_pred
        )

        # 7. Pillar 3: Privacy
        privacy_res = evaluate_privacy(
            df=df,
            train_acc=train_acc,
            test_acc=test_acc
        )

        # 8. Pillar 4: Robustness
        robustness_res = evaluate_robustness(
            df=df,
            feature_columns=feature_columns,
            model=model,
            y_pred=y_pred
        )

        # 9. Pillar 5: Transparency & Governance
        transparency_res = evaluate_transparency(
            df=df,
            metadata=metadata,
            model_name=model.__class__.__name__ if model else None,
            feature_columns=feature_columns,
            target_column=target_column
        )

        # Calculate Unified Weighted Composite Score
        f_score = fairness_res.get("score", 100)
        e_score = explainability_res.get("score", 100)
        p_score = privacy_res.get("score", 100)
        r_score = robustness_res.get("score", 100)
        t_score = transparency_res.get("score", 100)

        composite_score = (
            self.weights["fairness"] * f_score +
            self.weights["explainability"] * e_score +
            self.weights["privacy"] * p_score +
            self.weights["robustness"] * r_score +
            self.weights["transparency"] * t_score
        )
        composite_score = round(float(composite_score), 1)

        letter_grade = get_letter_grade(composite_score)
        overall_status = get_risk_status(composite_score)

        # Compile prioritized action items
        action_items = []

        for p_name, p_res, weight in [
            ("Fairness", fairness_res, self.weights["fairness"]),
            ("Privacy", privacy_res, self.weights["privacy"]),
            ("Robustness", robustness_res, self.weights["robustness"]),
            ("Explainability", explainability_res, self.weights["explainability"]),
            ("Transparency", transparency_res, self.weights["transparency"])
        ]:
            p_score_val = p_res.get("score", 100)
            p_recs = p_res.get("recommendations", [])
            p_findings = p_res.get("findings", [])

            severity = "LOW"
            if p_score_val < 50:
                severity = "CRITICAL"
            elif p_score_val < 70:
                severity = "HIGH"
            elif p_score_val < 85:
                severity = "MEDIUM"

            for rec in p_recs:
                action_items.append({
                    "pillar": p_name,
                    "severity": severity,
                    "score": p_score_val,
                    "recommendation": rec,
                    "context": p_findings[0] if p_findings else ""
                })

        # Sort actions by severity: CRITICAL > HIGH > MEDIUM > LOW
        severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        action_items.sort(key=lambda x: severity_order.get(x["severity"], 4))

        # Full structured audit report
        resolved_sensitive = fairness_res.get("sensitive_column") or (sensitive_column if sensitive_column else "Multi-Attribute Auto")

        report = {
            "summary": {
                "audit_timestamp": audit_timestamp,
                "overall_score": composite_score,
                "letter_grade": letter_grade,
                "status": overall_status,
                "dataset_rows": len(df),
                "dataset_columns": len(df.columns),
                "target_column": target_column,
                "sensitive_column": resolved_sensitive,
                "detected_demographics": fairness_res.get("detected_demographics", []),
                "model_evaluated": model is not None or y_pred is not None,
                "weights_used": self.weights
            },
            "pillar_scores": {
                "fairness": f_score,
                "explainability": e_score,
                "privacy": p_score,
                "robustness": r_score,
                "transparency": t_score
            },
            "pillars": {
                "fairness": fairness_res,
                "explainability": explainability_res,
                "privacy": privacy_res,
                "robustness": robustness_res,
                "transparency": transparency_res
            },
            "prioritized_actions": action_items
        }

        return report
