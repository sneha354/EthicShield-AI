"""
AI Ethics Auditor Command Line Interface (CLI)
Enables automated audits of datasets and ML models directly from terminal or CI/CD pipelines.
"""

import argparse
import sys
import os
import joblib
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.engine.auditor import AIEthicsAuditor
from backend.samples.generator import get_scenario, list_scenarios
from backend.reports.generator import generate_html_report, export_json_report


def main():
    parser = argparse.ArgumentParser(description="EthicShield AI - AI Ethics & Safety Auditor CLI")
    parser.add_argument("--dataset", type=str, help="Path to input dataset (CSV, JSON, or Parquet)")
    parser.add_argument("--model", type=str, help="Path to serialized ML model (.pkl or .joblib)")
    parser.add_argument("--sensitive-column", type=str, help="Name of protected/sensitive attribute column")
    parser.add_argument("--target", type=str, help="Name of target label column")
    parser.add_argument("--privileged-group", type=str, help="Value of privileged group in sensitive column")
    parser.add_argument("--sample", type=str, choices=["credit_lending", "hr_recruitment", "healthcare_readmission"], help="Run one of the pre-loaded benchmark audit scenarios")
    parser.add_argument("--output", type=str, default="audit_report.html", help="Path to save HTML report output")
    parser.add_argument("--json-output", type=str, help="Path to save JSON report artifact")
    parser.add_argument("--list-samples", action="store_true", help="List available sample scenarios")

    args = parser.parse_args()

    if args.list_samples:
        print("Available benchmark scenarios:")
        for s in list_scenarios():
            print(f"  * {s['id']:<24}: {s['title']} ({s['rows']} rows, target: '{s['target_column']}')")
        return

    auditor = AIEthicsAuditor()

    if args.sample:
        print(f"\n[*] Running AI Ethics Audit on benchmark scenario: '{args.sample}'...")
        df, model, meta = get_scenario(args.sample)
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
    elif args.dataset:
        if not os.path.exists(args.dataset):
            print(f"Error: Dataset file not found at {args.dataset}")
            sys.exit(1)

        print(f"\n[*] Loading dataset from {args.dataset}...")
        if args.dataset.endswith(".csv"):
            df = pd.read_csv(args.dataset)
        elif args.dataset.endswith(".json"):
            df = pd.read_json(args.dataset)
        elif args.dataset.endswith(".parquet"):
            df = pd.read_parquet(args.dataset)
        else:
            df = pd.read_csv(args.dataset)

        model = None
        if args.model:
            if not os.path.exists(args.model):
                print(f"Error: Model file not found at {args.model}")
                sys.exit(1)
            print(f"[*] Loading model from {args.model}...")
            model = joblib.load(args.model)

        print("[*] Executing multi-pillar AI Ethics Audit...")
        report = auditor.audit(
            df=df,
            sensitive_column=args.sensitive_column,
            target_column=args.target,
            privileged_group=args.privileged_group,
            model=model
        )
    else:
        parser.print_help()
        print("\nTip: Run with '--sample credit_lending' for an instant demonstration audit.")
        return

    # Print Terminal Summary
    summary = report["summary"]
    scores = report["pillar_scores"]
    print("\n" + "=" * 65)
    print("           AI ETHICS & SAFETY AUDIT EXECUTIVE SUMMARY           ")
    print("=" * 65)
    print(f"Overall Ethics Score : {summary['overall_score']:>5.1f} / 100  (Grade: {summary['letter_grade']})")
    print(f"Status               : {summary['status']}")
    print(f"Total Rows Audited   : {summary['dataset_rows']}")
    print("-" * 65)
    print("PILLAR BREAKDOWN:")
    print(f"  1. Fairness & Bias     : {scores['fairness']:>3}/100  (Risk: {report['pillars']['fairness']['risk_level']})")
    print(f"  2. Explainability      : {scores['explainability']:>3}/100  (Risk: {report['pillars']['explainability']['risk_level']})")
    print(f"  3. Privacy Protection  : {scores['privacy']:>3}/100  (Risk: {report['pillars']['privacy']['risk_level']})")
    print(f"  4. Robustness & Safety : {scores['robustness']:>3}/100  (Risk: {report['pillars']['robustness']['risk_level']})")
    print(f"  5. Transparency        : {scores['transparency']:>3}/100  (Risk: {report['pillars']['transparency']['risk_level']})")
    print("-" * 65)
    print(f"PRIORITIZED ACTION ITEMS ({len(report['prioritized_actions'])} items):")
    for i, act in enumerate(report["prioritized_actions"][:4], 1):
        print(f"  [{act['severity']}] {act['pillar']}: {act['recommendation']}")
    if len(report["prioritized_actions"]) > 4:
        print(f"  ... and {len(report['prioritized_actions']) - 4} more items.")
    print("=" * 65)

    # Save outputs
    if args.output:
        html = generate_html_report(report)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"\n[+] Standalone HTML report saved to: {os.path.abspath(args.output)}")

    if args.json_output:
        json_content = export_json_report(report)
        with open(args.json_output, "w", encoding="utf-8") as f:
            f.write(json_content)
        print(f"[+] Machine-readable JSON report saved to: {os.path.abspath(args.json_output)}")


if __name__ == "__main__":
    main()

