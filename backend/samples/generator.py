"""
Sample Data & Model Generator for AI Ethics Auditor
Creates realistic, multi-scenario benchmark datasets and trained ML models:
1. Credit Lending Risk (Demographic Disparity & Bias)
2. HR Candidate Screening (Gender Disparity + PII Exposure)
3. Healthcare Patient Readmission (Clinical Disparity & Noise Sensitivity)
"""

import os
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import joblib


SAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SAMPLES_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)


def generate_credit_lending_scenario() -> Tuple[pd.DataFrame, Any, Dict[str, Any]]:
    """
    Scenario 1: Credit Lending Risk
    Fairness challenge: Gender disparity violating the 80% rule.
    """
    np.random.seed(42)
    n = 800

    gender = np.random.choice(["Male", "Female"], size=n, p=[0.60, 0.40])
    age = np.random.randint(21, 68, size=n)
    income = np.random.normal(55000, 18000, size=n).clip(18000, 160000).round()
    loan_amount = np.random.normal(18000, 8000, size=n).clip(2000, 60000).round()
    credit_score = np.random.normal(670, 65, size=n).clip(350, 850).round()
    years_employed = np.random.exponential(5, size=n).clip(0, 35).round(1)
    debt_to_income = (np.random.beta(2, 5, size=n) * 60).round(1)

    # Underlying approval probability with slight historical systemic bias
    # Females have lower approval probability despite similar income
    z = (
        0.005 * (credit_score - 650) +
        0.00003 * (income - 40000) -
        0.00004 * (loan_amount - 15000) +
        0.05 * (years_employed - 4) -
        0.03 * (debt_to_income - 25) +
        np.where(gender == "Male", 0.45, -0.35)  # Historical bias injection
    )
    prob = 1.0 / (1.0 + np.exp(-z))
    approved = (np.random.rand(n) < prob).astype(int)

    df = pd.DataFrame({
        "gender": gender,
        "age": age,
        "income": income,
        "loan_amount": loan_amount,
        "credit_score": credit_score,
        "years_employed": years_employed,
        "debt_to_income": debt_to_income,
        "approved": approved
    })

    # Train a realistic Random Forest model
    features = ["age", "income", "loan_amount", "credit_score", "years_employed", "debt_to_income"]
    X = df[features].copy()
    y = df["approved"].values

    model = RandomForestClassifier(n_estimators=60, max_depth=6, random_state=42)
    model.fit(X, y)

    meta = {
        "scenario_id": "credit_lending",
        "title": "Credit Lending Approval Audit",
        "domain": "Financial Services / Lending",
        "description": "Evaluates loan approval algorithms for Fair Housing Act / ECOA compliance. Exhibits demographic disparate impact across gender.",
        "sensitive_column": "gender",
        "privileged_group": "Male",
        "target_column": "approved",
        "feature_columns": features,
        "model_name": "CreditRiskRandomForest",
        "intended_use": "Automated initial credit assessment for unsecured personal loans.",
        "out_of_scope_uses": "Sole basis for denying high-value mortgages without human review.",
        "license": "Proprietary / Internal Financial",
        "data_provenance": "Historical credit bureau records (2021-2023)",
        "train_acc": 0.88,
        "test_acc": 0.81
    }

    return df, model, meta


def generate_hr_recruitment_scenario() -> Tuple[pd.DataFrame, Any, Dict[str, Any]]:
    """
    Scenario 2: HR Candidate Screening
    Challenges: Gender disparity + Embedded PII (Names, Emails, Phone Numbers).
    """
    np.random.seed(123)
    n = 600

    genders = np.random.choice(["Male", "Female", "Non-Binary"], size=n, p=[0.55, 0.40, 0.05])
    male_names = ["James Smith", "Michael Brown", "David Johnson", "Robert Davis", "John Miller", "William Wilson"]
    female_names = ["Emily Clark", "Sarah Taylor", "Jessica White", "Ashley Harris", "Amanda Martin", "Megan Moore"]
    nb_names = ["Alex Morgan", "Jordan Taylor", "Taylor Lee", "Morgan Kelly", "Casey Jordan"]

    candidate_names = []
    emails = []
    phones = []

    for g in genders:
        if g == "Male":
            name = np.random.choice(male_names)
        elif g == "Female":
            name = np.random.choice(female_names)
        else:
            name = np.random.choice(nb_names)
        
        first = name.split()[0].lower()
        candidate_names.append(f"{name} {np.random.randint(10, 99)}")
        emails.append(f"{first}.{np.random.randint(100, 999)}@example-talent.com")
        phones.append(f"+1-{np.random.randint(200, 999)}-{np.random.randint(100, 999)}-{np.random.randint(1000, 9999)}")

    years_experience = np.random.exponential(4, size=n).clip(0, 25).round(1)
    skill_score = np.random.normal(72, 14, size=n).clip(30, 100).round()
    certifications_count = np.random.poisson(1.5, size=n).clip(0, 6)
    leadership_score = np.random.normal(65, 15, size=n).clip(20, 100).round()
    education_level = np.random.choice([1, 2, 3], size=n, p=[0.4, 0.45, 0.15])  # 1: Bachelor, 2: Master, 3: PhD

    # Biased callback decision
    z = (
        0.08 * (skill_score - 70) +
        0.20 * (years_experience - 4) +
        0.04 * (leadership_score - 60) +
        0.30 * (education_level - 1) +
        np.where(genders == "Male", 0.50, -0.40)  # Bias against female/non-binary candidates
    )
    prob = 1.0 / (1.0 + np.exp(-z))
    callback = (np.random.rand(n) < prob).astype(int)

    df = pd.DataFrame({
        "candidate_name": candidate_names,
        "email": emails,
        "phone": phones,
        "gender": genders,
        "years_experience": years_experience,
        "skill_score": skill_score,
        "certifications_count": certifications_count,
        "leadership_score": leadership_score,
        "education_level": education_level,
        "interview_callback": callback
    })

    features = ["years_experience", "skill_score", "certifications_count", "leadership_score", "education_level"]
    X = df[features].copy()
    y = df["interview_callback"].values

    model = GradientBoostingClassifier(n_estimators=50, max_depth=4, random_state=123)
    model.fit(X, y)

    meta = {
        "scenario_id": "hr_recruitment",
        "title": "HR Resume Screening & Recruitment Audit",
        "domain": "Human Resources / Talent Acquisition",
        "description": "Audits candidate ranking systems. Contains direct PII leakage (names, emails, phone numbers) and demographic disparities.",
        "sensitive_column": "gender",
        "privileged_group": "Male",
        "target_column": "interview_callback",
        "feature_columns": features,
        "model_name": "ResumeScreeningGradientBooster",
        "intended_use": "Preliminary resume prioritization for engineering and analyst roles.",
        "out_of_scope_uses": "Automated rejection of candidates without human recruiter review.",
        "license": "Internal Enterprise HR",
        "data_provenance": "Applicant Tracking System (ATS) 2022-2023 logs",
        "train_acc": 0.94,
        "test_acc": 0.76  # Notice high generalization gap indicating potential memorization / MIA risk!
    }

    return df, model, meta


def generate_healthcare_scenario() -> Tuple[pd.DataFrame, Any, Dict[str, Any]]:
    """
    Scenario 3: Healthcare Patient Readmission
    Challenges: Age disparity, noise sensitivity, outlier profiling.
    """
    np.random.seed(999)
    n = 700

    age_group = np.random.choice(["Young (18-40)", "Middle-Aged (41-65)", "Elderly (66+)"], size=n, p=[0.25, 0.45, 0.30])
    bmi = np.random.normal(28.5, 5.2, size=n).clip(16.0, 52.0).round(1)
    blood_pressure = np.random.normal(128, 18, size=n).clip(85, 210).round()
    glucose_level = np.random.normal(115, 35, size=n).clip(65, 340).round()
    prior_admissions = np.random.poisson(0.8, size=n).clip(0, 7)
    days_in_hospital = np.random.geometric(0.25, size=n).clip(1, 20)

    # Risk of readmission
    z = (
        0.03 * (blood_pressure - 120) +
        0.015 * (glucose_level - 100) +
        0.35 * prior_admissions +
        0.12 * (days_in_hospital - 3) +
        np.where(age_group == "Elderly (66+)", 0.6, -0.2)
    )
    prob = 1.0 / (1.0 + np.exp(-z))
    readmitted = (np.random.rand(n) < prob).astype(int)

    df = pd.DataFrame({
        "age_group": age_group,
        "bmi": bmi,
        "blood_pressure": blood_pressure,
        "glucose_level": glucose_level,
        "prior_admissions": prior_admissions,
        "days_in_hospital": days_in_hospital,
        "readmitted_30d": readmitted
    })

    features = ["bmi", "blood_pressure", "glucose_level", "prior_admissions", "days_in_hospital"]
    X = df[features].copy()
    y = df["readmitted_30d"].values

    model = LogisticRegression(max_iter=500, random_state=999)
    model.fit(X, y)

    meta = {
        "scenario_id": "healthcare_readmission",
        "title": "Hospital Readmission Clinical Predictor",
        "domain": "Healthcare / Clinical Decision Support",
        "description": "Evaluates 30-day hospital readmission predictions for clinical safety, age group fairness, and noise sensitivity.",
        "sensitive_column": "age_group",
        "privileged_group": "Middle-Aged (41-65)",
        "target_column": "readmitted_30d",
        "feature_columns": features,
        "model_name": "ClinicalReadmissionLogisticModel",
        "intended_use": "Flagging discharged patients who may benefit from telephone follow-up care.",
        "out_of_scope_uses": "Altering inpatient medication or surgical treatment plans autonomously.",
        "license": "Research & Clinical Trial (HIPAA Compliant)",
        "data_provenance": "EHR Anonymized Inpatient Discharge Cohort 2023",
        "train_acc": 0.82,
        "test_acc": 0.80
    }

    return df, model, meta


SCENARIO_GENERATORS = {
    "credit_lending": generate_credit_lending_scenario,
    "hr_recruitment": generate_hr_recruitment_scenario,
    "healthcare_readmission": generate_healthcare_scenario
}


def get_scenario(scenario_id: str) -> Tuple[pd.DataFrame, Any, Dict[str, Any]]:
    if scenario_id not in SCENARIO_GENERATORS:
        raise ValueError(f"Unknown scenario ID: '{scenario_id}'. Available: {list(SCENARIO_GENERATORS.keys())}")
    return SCENARIO_GENERATORS[scenario_id]()


def list_scenarios() -> list:
    scenarios = []
    for sid, fn in SCENARIO_GENERATORS.items():
        df, model, meta = fn()
        scenarios.append({
            "id": sid,
            "title": meta["title"],
            "domain": meta["domain"],
            "description": meta["description"],
            "rows": len(df),
            "columns": len(df.columns),
            "sensitive_column": meta["sensitive_column"],
            "target_column": meta["target_column"],
            "model_name": meta["model_name"]
        })
    return scenarios

