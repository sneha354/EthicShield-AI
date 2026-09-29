"""
Fairness and Bias Audit Engine
Evaluates dataset representation disparity and ML model algorithmic fairness metrics:
- Autonomous Multi-Attribute Demographic Discovery
- Disparate Impact (80% / Four-Fifths rule)
- Demographic Parity Difference
- Equal Opportunity Difference (True Positive Rate Parity)
- Equalized Odds Difference
- Subgroup performance breakdown (Accuracy, Precision, Recall, FPR, FNR)
- Intersectional & Multi-Demographic Scorecards
"""

from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd

try:
    from fairlearn.metrics import (
        MetricFrame,
        selection_rate,
        demographic_parity_difference,
        demographic_parity_ratio,
        equalized_odds_difference,
        true_positive_rate,
        false_positive_rate
    )
    FAIRLEARN_AVAILABLE = True
except Exception:
    FAIRLEARN_AVAILABLE = False


DEMOGRAPHIC_KEYWORDS = [
    "gender", "sex", "age", "race", "ethnicity", "religion",
    "marital", "nationality", "disability", "foreign", "citizen",
    "personal_status"
]

DEMOGRAPHIC_CATEGORIES = {
    "male", "female", "man", "woman", "white", "black", "asian",
    "hispanic", "latino", "married", "single", "divorced", "widowed"
}


def auto_detect_demographics(df: pd.DataFrame, target_column: Optional[str] = None) -> List[str]:
    """Auto-detect all demographic / protected attribute candidate columns in a dataset."""
    candidates = []
    for col in df.columns:
        if col == target_column:
            continue
        c_lower = col.lower()
        if any(k in c_lower for k in DEMOGRAPHIC_KEYWORDS):
            candidates.append(col)
        elif df[col].dtype == object or str(df[col].dtype) == "category":
            unique_vals = set(df[col].dropna().astype(str).str.lower().unique())
            if unique_vals.intersection(DEMOGRAPHIC_CATEGORIES):
                if col not in candidates:
                    candidates.append(col)
    return candidates


def evaluate_single_attribute(
    df: pd.DataFrame,
    sensitive_column: str,
    target_column: Optional[str] = None,
    y_true: Optional[Union[np.ndarray, pd.Series, List]] = None,
    y_pred: Optional[Union[np.ndarray, pd.Series, List]] = None,
    privileged_group: Optional[Any] = None
) -> Dict[str, Any]:
    """Evaluates fairness metrics for a single protected attribute column."""
    results: Dict[str, Any] = {
        "sensitive_column": sensitive_column,
        "privileged_group": privileged_group,
        "dataset_fairness": {},
        "model_fairness": {},
        "subgroup_metrics": [],
        "score": 100,
        "risk_level": "LOW",
        "findings": [],
        "recommendations": []
    }

    if sensitive_column not in df.columns:
        results["findings"].append(f"Sensitive column '{sensitive_column}' not found in dataset.")
        results["score"] = 50
        results["risk_level"] = "UNKNOWN"
        return results

    # If column is continuous numeric (like age), bin into standard legal demographic cohorts
    is_numeric = pd.api.types.is_numeric_dtype(df[sensitive_column])
    if is_numeric and df[sensitive_column].nunique() > 5:
        col_vals = df[sensitive_column].dropna()
        if "age" in sensitive_column.lower():
            sens_series = pd.cut(
                df[sensitive_column],
                bins=[-np.inf, 25, 50, np.inf],
                labels=["Young (<25)", "Prime (25-50)", "Senior (>50)"]
            ).astype(str)
        else:
            sens_series = pd.qcut(
                df[sensitive_column],
                q=3,
                labels=["Low Cohort", "Mid Cohort", "High Cohort"],
                duplicates="drop"
            ).astype(str)
    else:
        sens_series = df[sensitive_column].astype(str)

    groups = [str(g) for g in sens_series.unique().tolist() if str(g).lower() not in ["nan", "none"]]
    results["groups"] = groups

    # 1. Dataset Representation Analysis
    total_count = len(df)
    group_counts = sens_series.value_counts().to_dict()
    representation = {str(g): {"count": int(c), "percentage": round((c / total_count) * 100, 2)} for g, c in group_counts.items()}
    results["dataset_fairness"]["representation"] = representation

    counts_list = list(group_counts.values())
    max_c, min_c = max(counts_list), min(counts_list)
    representation_ratio = min_c / max_c if max_c > 0 else 1.0
    results["dataset_fairness"]["representation_ratio"] = round(float(representation_ratio), 3)

    if representation_ratio < 0.3:
        results["findings"].append(
            f"Severe representation imbalance in '{sensitive_column}': least represented group has ratio {round(representation_ratio, 2)} relative to majority group."
        )
        results["recommendations"].append(f"Collect additional data or apply synthetic re-balancing for minority cohorts in '{sensitive_column}'.")
    elif representation_ratio < 0.6:
        results["findings"].append(
            f"Moderate dataset imbalance in '{sensitive_column}': ratio is {round(representation_ratio, 2)} relative to majority group."
        )

    # Historical base rates if target column exists
    if target_column and target_column in df.columns:
        y_target = df[target_column]
        if y_target.nunique() == 2:
            pos_label = 1 if 1 in y_target.unique() else y_target.unique()[0]
            base_rates = {}
            for g in groups:
                sub_y = y_target[sens_series == g]
                if len(sub_y) > 0:
                    pos_rate = float((sub_y == pos_label).mean())
                    base_rates[g] = round(pos_rate, 3)
            results["dataset_fairness"]["historical_base_rates"] = base_rates

            if len(base_rates) > 1:
                br_values = list(base_rates.values())
                max_br, min_br = max(br_values), min(br_values)
                br_diff = round(max_br - min_br, 3)
                results["dataset_fairness"]["base_rate_disparity"] = br_diff
                if br_diff > 0.15:
                    results["findings"].append(
                        f"Historical ground-truth labels show {round(br_diff*100, 1)}% base rate gap across '{sensitive_column}' groups."
                    )

    # 2. Model Fairness Metrics
    if y_pred is not None:
        y_pred = np.array(y_pred)
        if y_true is not None:
            y_true = np.array(y_true)

        unique_preds = np.unique(y_pred)
        pos_val = 1 if 1 in unique_preds else unique_preds[-1]

        subgroup_data = []
        sel_rates = {}
        tpr_rates = {}
        fpr_rates = {}
        accuracies = {}

        for g in groups:
            mask = (sens_series == g).values
            n_g = int(np.sum(mask))
            if n_g == 0:
                continue

            g_pred = y_pred[mask]
            sel_rate = float((g_pred == pos_val).mean())
            sel_rates[g] = sel_rate

            metric_entry = {
                "group": str(g),
                "count": n_g,
                "selection_rate": round(sel_rate, 4),
                "positive_count": int(np.sum(g_pred == pos_val))
            }

            if y_true is not None:
                g_true = y_true[mask]
                acc = float((g_pred == g_true).mean())
                accuracies[g] = acc
                metric_entry["accuracy"] = round(acc, 4)

                actual_pos = (g_true == pos_val)
                actual_neg = (g_true != pos_val)

                if np.sum(actual_pos) > 0:
                    tpr = float((g_pred[actual_pos] == pos_val).mean())
                    tpr_rates[g] = tpr
                    metric_entry["true_positive_rate"] = round(tpr, 4)
                    metric_entry["false_negative_rate"] = round(1.0 - tpr, 4)
                else:
                    metric_entry["true_positive_rate"] = None

                if np.sum(actual_neg) > 0:
                    fpr = float((g_pred[actual_neg] == pos_val).mean())
                    fpr_rates[g] = fpr
                    metric_entry["false_positive_rate"] = round(fpr, 4)
                else:
                    metric_entry["false_positive_rate"] = None

            subgroup_data.append(metric_entry)

        results["subgroup_metrics"] = subgroup_data

        # Auto-detect Privileged Group if not provided:
        # In adverse impact auditing, the privileged reference baseline is the group with the highest selection rate
        if not privileged_group or str(privileged_group).lower() == "auto":
            if sel_rates:
                privileged_group = max(sel_rates, key=sel_rates.get)
            else:
                privileged_group = max(group_counts, key=group_counts.get)
        results["privileged_group"] = str(privileged_group)

        # Disparate Impact Ratio & Demographic Parity Difference
        if len(sel_rates) > 1:
            rates = list(sel_rates.values())
            max_sr, min_sr = max(rates), min(rates)
            demographic_parity_diff = max_sr - min_sr

            priv_sr = sel_rates.get(str(privileged_group), max_sr)
            if priv_sr > 0:
                other_rates = [r for g, r in sel_rates.items() if str(g) != str(privileged_group)]
                lowest_unpriv_sr = min(other_rates) if other_rates else min_sr
                disparate_impact = lowest_unpriv_sr / priv_sr
            else:
                disparate_impact = (min_sr / max_sr) if max_sr > 0 else 1.0

            results["model_fairness"]["demographic_parity_difference"] = round(float(demographic_parity_diff), 4)
            results["model_fairness"]["disparate_impact_ratio"] = round(float(disparate_impact), 4)
            results["model_fairness"]["four_fifths_rule_passed"] = bool(disparate_impact >= 0.80)
            results["model_fairness"]["privileged_baseline"] = str(privileged_group)

            if disparate_impact < 0.80:
                results["findings"].append(
                    f"Violation of Four-Fifths (80%) Rule on '{sensitive_column}': Disparate Impact is {round(disparate_impact, 3)} "
                    f"(< 0.80 threshold vs. baseline '{privileged_group}')."
                )
                results["recommendations"].append(
                    f"Calibrate classification decision thresholds across '{sensitive_column}' to equalize favorable selection rates."
                )
            elif disparate_impact < 0.90:
                results["findings"].append(
                    f"Disparate Impact on '{sensitive_column}' is borderline ({round(disparate_impact, 3)} vs. '{privileged_group}')."
                )

        # Equal Opportunity (TPR disparity)
        if len(tpr_rates) > 1:
            tprs = list(tpr_rates.values())
            eq_opp_diff = max(tprs) - min(tprs)
            results["model_fairness"]["equal_opportunity_difference"] = round(float(eq_opp_diff), 4)
            if eq_opp_diff > 0.10:
                results["findings"].append(
                    f"True Positive Rate varies by {round(eq_opp_diff * 100, 1)}% across '{sensitive_column}' groups (Equal Opportunity gap)."
                )

        # Equalized Odds Difference
        if len(tpr_rates) > 1 and len(fpr_rates) > 1:
            fprs = list(fpr_rates.values())
            fpr_diff = max(fprs) - min(fprs)
            eq_opp_val = results["model_fairness"].get("equal_opportunity_difference", 0.0)
            eq_odds_diff = max(eq_opp_val, fpr_diff)
            results["model_fairness"]["equalized_odds_difference"] = round(float(eq_odds_diff), 4)
            results["model_fairness"]["false_positive_rate_difference"] = round(float(fpr_diff), 4)

    # 3. Compute Normalized Fairness Score (0 - 100)
    score = 100.0
    if results["model_fairness"]:
        di = results["model_fairness"].get("disparate_impact_ratio", 1.0)
        dp_diff = results["model_fairness"].get("demographic_parity_difference", 0.0)
        eq_opp = results["model_fairness"].get("equal_opportunity_difference", 0.0)

        if di < 0.80:
            score -= (0.80 - di) * 60
        elif di < 0.90:
            score -= 5

        if dp_diff > 0.10:
            score -= min(30, dp_diff * 60)
        if eq_opp > 0.10:
            score -= min(25, eq_opp * 50)
    else:
        if representation_ratio < 0.5:
            score -= (0.5 - representation_ratio) * 50
        base_disparity = results["dataset_fairness"].get("base_rate_disparity", 0.0)
        if base_disparity > 0.15:
            score -= base_disparity * 30

    final_score = int(max(0, min(100, round(score))))
    results["score"] = final_score

    if final_score >= 85:
        results["risk_level"] = "LOW"
    elif final_score >= 70:
        results["risk_level"] = "MODERATE"
    elif final_score >= 50:
        results["risk_level"] = "HIGH"
    else:
        results["risk_level"] = "CRITICAL"

    return results


def evaluate_fairness(
    df: pd.DataFrame,
    sensitive_column: Optional[str] = None,
    target_column: Optional[str] = None,
    y_true: Optional[Union[np.ndarray, pd.Series, List]] = None,
    y_pred: Optional[Union[np.ndarray, pd.Series, List]] = None,
    privileged_group: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Autonomous Multi-Pillar Fairness Assessment.
    If sensitive_column is omitted or set to 'auto', it auto-detects ALL demographic
    attributes present in the dataset and computes a comprehensive holistic fairness audit.
    """
    detected_demographics = auto_detect_demographics(df, target_column)

    # Mode 1: Explicit single attribute requested
    if sensitive_column and sensitive_column.lower() != "auto" and sensitive_column in df.columns:
        primary_res = evaluate_single_attribute(
            df=df,
            sensitive_column=sensitive_column,
            target_column=target_column,
            y_true=y_true,
            y_pred=y_pred,
            privileged_group=privileged_group
        )
        # Also evaluate any other detected demographic columns for the multi-attribute scorecard
        breakdowns = {sensitive_column: primary_res}
        for col in detected_demographics:
            if col != sensitive_column:
                breakdowns[col] = evaluate_single_attribute(
                    df=df,
                    sensitive_column=col,
                    target_column=target_column,
                    y_true=y_true,
                    y_pred=y_pred,
                    privileged_group=None
                )
        primary_res["demographic_breakdowns"] = breakdowns
        primary_res["detected_demographics"] = list(breakdowns.keys())
        return primary_res

    # Mode 2: Autonomous Multi-Demographic Audit
    if detected_demographics:
        breakdowns = {}
        for col in detected_demographics:
            breakdowns[col] = evaluate_single_attribute(
                df=df,
                sensitive_column=col,
                target_column=target_column,
                y_true=y_true,
                y_pred=y_pred,
                privileged_group=None
            )

        # Select primary attribute based on highest risk (lowest disparate impact ratio / lowest score)
        def get_di(item):
            mf = item.get("model_fairness", {})
            return mf.get("disparate_impact_ratio", 1.0)

        primary_col = min(breakdowns.keys(), key=lambda k: (get_di(breakdowns[k]), breakdowns[k]["score"]))
        primary_res = dict(breakdowns[primary_col])

        # Compute holistic composite fairness score across ALL detected demographics
        all_scores = [b["score"] for b in breakdowns.values()]
        min_score = min(all_scores)
        avg_score = sum(all_scores) / len(all_scores)
        composite_fairness_score = int(round(0.6 * min_score + 0.4 * avg_score))

        # Consolidate findings and recommendations across all demographics
        all_findings = []
        all_recs = []
        for col, b in breakdowns.items():
            all_findings.extend(b.get("findings", []))
            all_recs.extend(b.get("recommendations", []))

        # Deduplicate recommendations
        unique_recs = list(dict.fromkeys(all_recs))
        if not unique_recs:
            unique_recs.append("All detected demographic attributes meet standard fairness thresholds.")

        primary_res["score"] = composite_fairness_score
        primary_res["demographic_breakdowns"] = breakdowns
        primary_res["detected_demographics"] = detected_demographics
        primary_res["findings"] = all_findings
        primary_res["recommendations"] = unique_recs
        primary_res["sensitive_column"] = primary_col
        primary_res["is_autonomous_multi_attribute"] = True

        if composite_fairness_score >= 85:
            primary_res["risk_level"] = "LOW"
        elif composite_fairness_score >= 70:
            primary_res["risk_level"] = "MODERATE"
        elif composite_fairness_score >= 50:
            primary_res["risk_level"] = "HIGH"
        else:
            primary_res["risk_level"] = "CRITICAL"

        return primary_res

    # Mode 3: No demographic columns detected
    return {
        "sensitive_column": "None Detected",
        "privileged_group": "None",
        "dataset_fairness": {},
        "model_fairness": {},
        "subgroup_metrics": [],
        "demographic_breakdowns": {},
        "detected_demographics": [],
        "score": 90,
        "risk_level": "LOW",
        "findings": [
            "Dataset does not contain explicit demographic or protected attribute columns (e.g. sex, age, race).",
            "Algorithmic disparity risk is minimized for direct protected attributes; continue auditing explainability and robustness."
        ],
        "recommendations": [
            "Ensure surrogate or proxy features (e.g., zip codes) are evaluated in the Explainability pillar for indirect bias."
        ]
    }
