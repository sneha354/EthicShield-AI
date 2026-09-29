"""
Transparency and Governance Audit Engine
Generates Model Cards (Mitchell et al.) and Dataset Datasheets (Gebru et al.),
evaluates documentation completeness, data provenance, and governance accountability.
"""

from typing import Dict, Any, List, Optional
import datetime
import pandas as pd


DEFAULT_CHECKLIST = [
    {"id": "intended_use", "title": "Intended Use Case Specified", "weight": 20, "description": "Clear statement of what domain and decisions the model is meant for."},
    {"id": "out_of_scope", "title": "Out-of-Scope Uses Documented", "weight": 15, "description": "Explicit declaration of scenarios where model use is prohibited or unvalidated."},
    {"id": "provenance", "title": "Data Provenance & Collection Method", "weight": 20, "description": "Source of data, collection timeframe, and sampling strategy."},
    {"id": "limitations", "title": "Known Limitations & Caveats", "weight": 15, "description": "Known blind spots, edge cases, or demographic representation gaps."},
    {"id": "license", "title": "License & Commercial Use Rights", "weight": 15, "description": "Clear software/data license (e.g. MIT, Apache-2.0, CC-BY-4.0)."},
    {"id": "human_oversight", "title": "Human-in-the-Loop Oversight Policy", "weight": 15, "description": "Process for human review of high-stakes or flagged decisions."}
]


def evaluate_transparency(
    df: pd.DataFrame,
    metadata: Optional[Dict[str, Any]] = None,
    model_name: Optional[str] = None,
    feature_columns: Optional[List[str]] = None,
    target_column: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run transparency audit, generate Model Card & Datasheet, and compute governance score.
    """
    metadata = metadata or {}
    results: Dict[str, Any] = {
        "model_card": {},
        "datasheet": {},
        "governance_checklist": [],
        "completeness_score": 0,
        "score": 100,
        "risk_level": "LOW",
        "findings": [],
        "recommendations": []
    }

    # 1. Dataset Datasheet (Gebru et al.)
    results["datasheet"] = {
        "dataset_name": metadata.get("dataset_name", "Uploaded Dataset"),
        "total_instances": len(df),
        "total_features": len(df.columns),
        "feature_list": list(df.columns),
        "missing_values_count": int(df.isna().sum().sum()),
        "collection_period": metadata.get("collection_period", "Not Specified"),
        "data_provenance": metadata.get("data_provenance", "Internal or User-Provided"),
        "license": metadata.get("license", "Not Specified"),
        "anonymization_performed": metadata.get("anonymization_performed", False)
    }

    # 2. Model Card (Mitchell et al.)
    results["model_card"] = {
        "model_name": model_name or metadata.get("model_name", "ML Classifier / Predictor"),
        "version": metadata.get("model_version", "1.0.0"),
        "creation_date": metadata.get("date", datetime.date.today().isoformat()),
        "intended_use": metadata.get("intended_use", "Assisting decision-making in target domain"),
        "out_of_scope_uses": metadata.get("out_of_scope_uses", "Fully autonomous high-stakes decisions without human review"),
        "target_variable": target_column or metadata.get("target_column", "Target"),
        "features_used": feature_columns or list(df.columns),
        "ethical_considerations": metadata.get("ethical_considerations", "Continuous monitoring for demographic parity is required.")
    }

    # 3. Governance Checklist & Completeness Evaluation
    checklist_results = []
    achieved_weight = 0
    total_weight = sum(item["weight"] for item in DEFAULT_CHECKLIST)

    for item in DEFAULT_CHECKLIST:
        c_id = item["id"]
        # Check if metadata provides explicit entry for this item
        val = metadata.get(c_id)
        passed = False
        if val and str(val).strip().lower() not in ["", "none", "not specified", "unknown", "n/a"]:
            passed = True
            achieved_weight += item["weight"]
        elif c_id == "human_oversight" and metadata.get("human_in_the_loop") is True:
            passed = True
            achieved_weight += item["weight"]
        elif c_id == "license" and metadata.get("license") and metadata.get("license") != "Not Specified":
            passed = True
            achieved_weight += item["weight"]

        checklist_results.append({
            "id": c_id,
            "title": item["title"],
            "description": item["description"],
            "weight": item["weight"],
            "status": "PASSED" if passed else "MISSING"
        })

    results["governance_checklist"] = checklist_results
    completeness_pct = int(round((achieved_weight / total_weight) * 100))
    results["completeness_score"] = completeness_pct

    # Deductions based on missing governance components
    score = completeness_pct

    missing_items = [item["title"] for item in checklist_results if item["status"] == "MISSING"]
    if missing_items:
        results["findings"].append(
            f"Transparency documentation incomplete ({completeness_pct}%). Missing governance items: {', '.join(missing_items[:3])}"
            + (" and others." if len(missing_items) > 3 else ".")
        )
        for item in missing_items[:3]:
            results["recommendations"].append(f"Document '{item}' in the model card / datasheet repository before production deployment.")

    final_score = int(max(0, min(100, round(score))))
    results["score"] = final_score

    if final_score >= 80:
        results["risk_level"] = "LOW"
    elif final_score >= 60:
        results["risk_level"] = "MODERATE"
    elif final_score >= 40:
        results["risk_level"] = "HIGH"
    else:
        results["risk_level"] = "CRITICAL"

    if not results["recommendations"]:
        results["recommendations"].append("Model Card and Datasheet meet transparency standards. Maintain version control on updates.")

    return results

