"""
Privacy and Data Protection Audit Engine
Evaluates PII (Personally Identifiable Information) exposure using Microsoft Presidio,
Quasi-Identifier Re-identification Risk (k-anonymity), and Model Membership Inference Risk.
"""

from typing import Dict, Any, List, Optional
import re
import numpy as np
import pandas as pd

try:
    from presidio_analyzer import AnalyzerEngine
    PRESIDIO_AVAILABLE = True
except Exception:
    PRESIDIO_AVAILABLE = False


# Fallback regex patterns if Presidio is unavailable or for supplementary fast scanning
PATTERNS = {
    "EMAIL_ADDRESS": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
    "PHONE_NUMBER": r"(\+?\d{1,3}[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}",
    "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    "IP_ADDRESS": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
    "SSN": r"\b\d{3}-\d{2}-\d{4}\b"
}


def mask_sample_text(text: str, entity_type: str) -> str:
    """Mask sensitive string for safe UI presentation."""
    s = str(text).strip()
    if len(s) <= 4:
        return "****"
    if "@" in s and entity_type == "EMAIL_ADDRESS":
        parts = s.split("@")
        return f"{parts[0][0]}***@{parts[1]}"
    return f"{s[:2]}****{s[-2:]}"


_ANALYZER_INSTANCE = None

def get_analyzer():
    global _ANALYZER_INSTANCE
    if _ANALYZER_INSTANCE is None and PRESIDIO_AVAILABLE:
        try:
            _ANALYZER_INSTANCE = AnalyzerEngine()
        except Exception:
            _ANALYZER_INSTANCE = None
    return _ANALYZER_INSTANCE


def evaluate_privacy(
    df: pd.DataFrame,
    quasi_identifiers: Optional[List[str]] = None,
    train_acc: Optional[float] = None,
    test_acc: Optional[float] = None,
    model_probabilities: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Run privacy audit covering PII detection, k-anonymity, and membership inference risk.
    """
    results: Dict[str, Any] = {
        "pii_detected": [],
        "total_pii_count": 0,
        "k_anonymity": {},
        "membership_inference_risk": {},
        "score": 100,
        "risk_level": "LOW",
        "findings": [],
        "recommendations": []
    }

    # 1. PII Scanning using Presidio Analyzer & Regex
    analyzer = get_analyzer()

    pii_findings: Dict[str, Dict[str, Any]] = {}
    sample_rows = min(150, len(df))
    df_sample = df.sample(n=sample_rows, random_state=42) if len(df) > sample_rows else df

    string_cols = [c for c in df.columns if df[c].dtype == object or pd.api.types.is_string_dtype(df[c])]

    for col in string_cols:
        col_name_lower = col.lower()
        # Direct column name heuristics
        if any(term in col_name_lower for term in ["email", "e-mail"]):
            key = f"{col}:EMAIL_ADDRESS"
            pii_findings[key] = {"column": col, "entity_type": "EMAIL_ADDRESS", "count": sample_rows, "confidence": 0.95, "sample": "user***@domain.com"}
        elif any(term in col_name_lower for term in ["phone", "mobile", "tel"]):
            key = f"{col}:PHONE_NUMBER"
            pii_findings[key] = {"column": col, "entity_type": "PHONE_NUMBER", "count": sample_rows, "confidence": 0.90, "sample": "+1-***-****"}
        elif any(term in col_name_lower for term in ["ssn", "social_security"]):
            key = f"{col}:SSN"
            pii_findings[key] = {"column": col, "entity_type": "SSN", "count": sample_rows, "confidence": 0.95, "sample": "***-**-****"}
        elif any(term in col_name_lower for term in ["first_name", "last_name", "candidate_name", "patient_name"]):
            key = f"{col}:PERSON"
            pii_findings[key] = {"column": col, "entity_type": "PERSON", "count": sample_rows, "confidence": 0.85, "sample": "J*** D***"}

        # Scan cell values
        for val in df_sample[col].dropna():
            val_str = str(val).strip()
            if not val_str or len(val_str) < 3:
                continue

            # Presidio inspection
            if analyzer is not None:
                try:
                    presidio_results = analyzer.analyze(text=val_str, language="en")
                    for p_res in presidio_results:
                        if p_res.score >= 0.5:
                            ent = p_res.entity_type
                            k = f"{col}:{ent}"
                            if k not in pii_findings:
                                pii_findings[k] = {
                                    "column": col,
                                    "entity_type": ent,
                                    "count": 0,
                                    "confidence": round(float(p_res.score), 2),
                                    "sample": mask_sample_text(val_str[p_res.start:p_res.end], ent)
                                }
                            pii_findings[k]["count"] += 1
                except Exception:
                    pass

            # Regex fallback
            for ent_name, pattern in PATTERNS.items():
                if re.search(pattern, val_str):
                    k = f"{col}:{ent_name}"
                    if k not in pii_findings:
                        pii_findings[k] = {
                            "column": col,
                            "entity_type": ent_name,
                            "count": 0,
                            "confidence": 0.85,
                            "sample": mask_sample_text(val_str, ent_name)
                        }
                    pii_findings[k]["count"] += 1

    pii_list = list(pii_findings.values())
    total_pii = sum(item["count"] for item in pii_list)
    results["pii_detected"] = pii_list
    results["total_pii_count"] = total_pii

    if total_pii > 0:
        entity_types = list({item["entity_type"] for item in pii_list})
        results["findings"].append(
            f"Detected {total_pii} potential PII instances across columns: {', '.join(set(item['column'] for item in pii_list))} "
            f"(Types: {', '.join(entity_types)})."
        )
        results["recommendations"].append(
            "Anonymize or redact all identified PII columns (using hashing, masking, or synthetic replacement) before dataset publication or training."
        )

    # 2. Quasi-Identifier & k-Anonymity Assessment
    # If not explicitly specified, detect potential quasi-identifiers
    detected_quasi = quasi_identifiers or []
    if not detected_quasi:
        potential_qi_terms = ["age", "gender", "sex", "zip", "postal", "city", "state", "job", "education", "ethnicity", "race"]
        detected_quasi = [c for c in df.columns if any(term in c.lower() for term in potential_qi_terms)][:4]

    if detected_quasi and len(detected_quasi) >= 2:
        try:
            equiv_classes = df.groupby(detected_quasi).size()
            k_min = int(equiv_classes.min())
            unique_rows = int((equiv_classes == 1).sum())
            pct_unique = round((unique_rows / len(df)) * 100, 2)
            pct_low_k = round(((equiv_classes < 3).sum() / len(equiv_classes)) * 100, 2)

            results["k_anonymity"] = {
                "quasi_identifiers": detected_quasi,
                "min_k": k_min,
                "unique_records_count": unique_rows,
                "unique_records_pct": pct_unique,
                "low_k_groups_pct": pct_low_k
            }

            if k_min == 1:
                results["findings"].append(
                    f"k-Anonymity violated (min k = 1) over quasi-identifiers {detected_quasi}. "
                    f"{pct_unique}% of records are uniquely distinguishable and vulnerable to linkage attacks."
                )
                results["recommendations"].append(
                    f"Apply data generalization (e.g. binning 'age' into 10-year ranges, truncating zip codes) to achieve at least k >= 5 anonymity."
                )
            elif k_min < 5:
                results["findings"].append(
                    f"Weak k-anonymity (min k = {k_min}) over quasi-identifiers {detected_quasi}. Some equivalence classes have fewer than 5 individuals."
                )
        except Exception:
            pass

    # 3. Model Membership Inference & Overfitting Risk
    if train_acc is not None and test_acc is not None:
        generalization_gap = max(0.0, float(train_acc - test_acc))
        results["membership_inference_risk"] = {
            "train_accuracy": round(float(train_acc), 4),
            "test_accuracy": round(float(test_acc), 4),
            "generalization_gap": round(generalization_gap, 4),
            "risk_assessment": "LOW" if generalization_gap < 0.08 else ("MODERATE" if generalization_gap < 0.18 else "HIGH")
        }

        if generalization_gap > 0.18:
            results["findings"].append(
                f"High Membership Inference Attack (MIA) vulnerability: Train-Test generalization gap is "
                f"{round(generalization_gap * 100, 1)}% (Train: {round(train_acc*100, 1)}%, Test: {round(test_acc*100, 1)}%). "
                f"Severe memorization allows adversaries to infer whether specific individuals were in the training set."
            )
            results["recommendations"].append(
                "Apply regularization (weight decay, dropout, or DP-SGD / differential privacy) to prevent training set memorization."
            )
        elif generalization_gap > 0.08:
            results["findings"].append(
                f"Moderate generalization gap ({round(generalization_gap * 100, 1)}%). Monitor for potential training data leakage."
            )

    # 4. Privacy Score Calculation (0 - 100)
    score = 100.0

    # Penalize detected PII
    if total_pii > 0:
        score -= min(45, total_pii * 4 + 10)

    # Penalize k-anonymity violations
    if results["k_anonymity"]:
        k_val = results["k_anonymity"]["min_k"]
        if k_val == 1:
            score -= min(25, results["k_anonymity"]["unique_records_pct"] * 0.5 + 10)
        elif k_val < 5:
            score -= 10

    # Penalize membership inference gap
    if results["membership_inference_risk"]:
        gap = results["membership_inference_risk"]["generalization_gap"]
        if gap > 0.18:
            score -= min(30, gap * 100)
        elif gap > 0.08:
            score -= 10

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
        results["recommendations"].append("No direct PII or critical re-identification vectors detected. Ensure adherence to regional privacy frameworks (GDPR/CCPA).")

    return results
