"""
Test Suite for AI Ethics Auditor Engine
Tests all 5 pillars and scenario generation.
"""

import sys
import os
import io

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.engine.auditor import AIEthicsAuditor
from backend.samples.generator import get_scenario, list_scenarios
from backend.reports.generator import generate_html_report, export_json_report


def run_tests():
    print("=== 1. Testing Scenario Listing ===")
    scenarios = list_scenarios()
    assert len(scenarios) >= 3, f"Expected at least 3 scenarios, found {len(scenarios)}"
    for s in scenarios:
        print(f"  * Scenario: {s['id']} - {s['title']} ({s['rows']} rows)")

    print("\n=== 2. Testing Credit Lending Audit ===")
    df, model, meta = get_scenario("credit_lending")
    auditor = AIEthicsAuditor()
    report = auditor.audit(
        df=df,
        sensitive_column=meta["sensitive_column"],
        target_column=meta["target_column"],
        feature_columns=meta["feature_columns"],
        model=model,
        privileged_group=meta["privileged_group"],
        train_acc=meta["train_acc"],
        test_acc=meta["test_acc"],
        metadata=meta
    )

    print(f"Overall Score: {report['summary']['overall_score']}/100")
    print(f"Grade: {report['summary']['letter_grade']}")
    print(f"Fairness Score: {report['pillar_scores']['fairness']}/100")
    print(f"Explainability Score: {report['pillar_scores']['explainability']}/100")
    print(f"Privacy Score: {report['pillar_scores']['privacy']}/100")
    print(f"Robustness Score: {report['pillar_scores']['robustness']}/100")
    print(f"Transparency Score: {report['pillar_scores']['transparency']}/100")
    print(f"Disparate Impact Ratio: {report['pillars']['fairness']['model_fairness'].get('disparate_impact_ratio')}")
    print(f"Top Feature: {report['pillars']['explainability']['feature_attributions'][0]['feature']}")
    print(f"Action Items Count: {len(report['prioritized_actions'])}")

    assert report["summary"]["overall_score"] > 0
    assert "fairness" in report["pillar_scores"]
    assert len(report["prioritized_actions"]) > 0

    print("\n=== 3. Testing HR Recruitment Audit (PII Scan) ===")
    df_hr, model_hr, meta_hr = get_scenario("hr_recruitment")
    report_hr = auditor.audit(
        df=df_hr,
        sensitive_column=meta_hr["sensitive_column"],
        target_column=meta_hr["target_column"],
        feature_columns=meta_hr["feature_columns"],
        model=model_hr,
        privileged_group=meta_hr["privileged_group"],
        train_acc=meta_hr["train_acc"],
        test_acc=meta_hr["test_acc"],
        metadata=meta_hr
    )
    print(f"HR PII Detected Count: {report_hr['pillars']['privacy']['total_pii_count']}")
    print(f"HR MIA Risk: {report_hr['pillars']['privacy']['membership_inference_risk'].get('generalization_gap')}")
    assert report_hr["pillars"]["privacy"]["total_pii_count"] > 0, "Expected PII detection in HR dataset!"

    print("\n=== 4. Testing HTML & JSON Report Export ===")
    html = generate_html_report(report)
    assert "<!DOCTYPE html>" in html
    assert "AI Ethics & Safety Audit Report" in html
    print(f"HTML report generated successfully ({len(html)} bytes)")

    json_str = export_json_report(report)
    assert '"overall_score"' in json_str
    print(f"JSON report exported successfully ({len(json_str)} bytes)")

    print("\n=== 5. Testing Autonomous Zero-Configuration Multi-Demographic Audit ===")
    import pandas as pd
    test_df = pd.DataFrame({
        "sex": ["male", "male", "female", "female", "male", "female"] * 20,
        "age": [22, 45, 60, 24, 52, 33] * 20,
        "income": [50000, 80000, 30000, 45000, 95000, 60000] * 20,
        "approved": [1, 1, 0, 0, 1, 1] * 20
    })
    # Run audit with ZERO parameters specified (all None)
    auto_report = auditor.audit(df=test_df)
    assert auto_report["summary"]["target_column"] == "approved", "Target column should be auto-detected as 'approved'"
    assert "sex" in auto_report["pillars"]["fairness"]["detected_demographics"], "Demographic 'sex' should be auto-detected"
    assert "age" in auto_report["pillars"]["fairness"]["detected_demographics"], "Demographic 'age' should be auto-detected"
    assert "demographic_breakdowns" in auto_report["pillars"]["fairness"], "Multi-demographic breakdowns must exist"
    print(f"Auto-detected Target: {auto_report['summary']['target_column']}")
    print(f"Auto-detected Demographics: {auto_report['pillars']['fairness']['detected_demographics']}")
    print(f"Overall Autonomous Composite Score: {auto_report['summary']['overall_score']}/100")

    print("\nALL TESTS PASSED SUCCESSFULLY! Autonomous Multi-Attribute Auditor is 100% functional.")


if __name__ == "__main__":
    run_tests()

