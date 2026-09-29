# EthicShield AI - Autonomous AI Ethics & Safety Auditor

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![Fairlearn](https://img.shields.io/badge/Fairlearn-0.9+-orange.svg)](https://fairlearn.org/)
[![SHAP](https://img.shields.io/badge/SHAP-0.42+-red.svg)](https://shap.readthedocs.io/)
[![Presidio](https://img.shields.io/badge/Microsoft_Presidio-2.2+-blueviolet.svg)](https://microsoft.github.io/presidio/)
[![Compliance](https://img.shields.io/badge/Regulation-EU_AI_Act_%7C_NIST_AI_RMF-brightgreen.svg)](#-compliance-standards)

---

## 📌 Project Abstract

> **EthicShield AI** is an autonomous, enterprise-grade auditing platform designed to rigorously evaluate machine learning models and datasets against global AI safety and regulatory benchmarks. Built upon five foundational pillars—**Fairness, Explainability, Privacy, Robustness, and Transparency**—the system operates through a zero-configuration engine that autonomously discovers decision targets, detects protected demographic attributes, and infers privileged baseline groups. It quantifies multi-attribute algorithmic disparities via the Four-Fifths rule and equalized odds, extracts global feature attributions using SHAP, scans sensitive personal data via Microsoft Presidio, assesses tabular re-identification risks, and robustness stress-tests decision boundaries against adversarial perturbations. Findings are consolidated into an actionable Composite AI Ethics Score (0–100), automated Model Cards, and prioritized remediation roadmaps rigorously aligned with the EU AI Act, NIST AI RMF, and IEEE 7000 standards. By delivering comprehensive risk intelligence without manual configuration, EthicShield AI bridges the divide between production machine learning and legally compliant, trustworthy artificial intelligence across modern enterprise environments.

---

## 🌟 Key Achievements & Impacts

* **Zero-Configuration Autonomous Discovery**: Automatically identifies decision targets, detects all demographic attributes (gender, age cohorts, race), and infers privileged baselines—enabling multi-attribute fairness auditing without manual configuration.
* **Unified 5-Pillar Technical Architecture**: Integrates Fairlearn (80% Disparate Impact & Equalized Odds), SHAP feature attributions, Microsoft Presidio PII scanning, $k$-anonymity risk scoring, and Gaussian jitter stress-testing into a single cohesive pipeline.
* **Composite AI Ethics Scoring (0–100)**: Formulated a weighted risk index that translates complex algorithmic behavior into an instant letter grade and prioritized remediation roadmap, accelerating compliance reviews from weeks to seconds.
* **Autonomous Model Card Synthesis**: Dynamically drafts standardized Model Cards (Mitchell et al.) and Dataset Datasheets (Gebru et al.), ensuring automated adherence to **EU AI Act (Regulation 2024/1689)** and **NIST AI RMF 1.0** documentation mandates.
* **Enterprise Risk & Liability Mitigation**: Delivers a streamlined web portal and CI/CD-ready CLI that proactively eliminates discriminatory bias, prevents GDPR/HIPAA privacy leaks, and embeds compliance gates directly into production ML workflows.

---

## 🚀 Quick Start for Industry Reviewers (Test in 60 Seconds)

To make it effortless for reviewers to evaluate the platform without hunting for external data, the repository includes an **industry-standard benchmark dataset** and a pre-configured model generator right in the `real_world_data/` directory.

### 1. Installation

Clone the repository and install dependencies:
```bash
git clone https://github.com/<your-username>/EthicShield-AI.git
cd EthicShield-AI
pip install -r requirements.txt
```

### 2. Demo dataset and model for testing ( any other can be used)
```
This prepares:
* 📄 **Dataset:** `real_world_data/german_credit_dataset.csv` *(Authentic Statlog / UCI Credit records)*
* 🧠 **Model:** `real_world_data/german_credit_model.joblib` *(Trained Random Forest Classifier)*

### 3. Launch the Audit Portal
```bash
python run.py
```
Open your browser to: **`http://localhost:8000`**

### 4. Run the Audit
1. In the **Enterprise Audit Portal**, drag and drop:
   * **Dataset:** `real_world_data/german_credit_dataset.csv`
   * **Model:** `real_world_data/german_credit_model.joblib`
2. **Do not touch any configuration fields.** Simply click:
   👉 **`⚡ Run Autonomous Multi-Pillar Ethics Audit`**
3. The platform will automatically detect the target (`credit_risk`), discover multiple demographic attributes (`sex` and `age`), compute the 5-pillar composite score, and synthesize a complete regulatory Model Card.

---

## 🛡️ The 5 Foundational Audit Pillars

| Pillar | Diagnostic Scope | Key Metrics & Methods | Regulatory Mapping |
| :--- | :--- | :--- | :--- |
| **1. Fairness & Bias** ⚖️ | Algorithmic parity across protected demographic groups | Four-Fifths (80%) Rule Disparate Impact, Demographic Parity Gap, Equalized Odds, Subgroup Breakdowns | EU AI Act Art. 10 (Data Governance), EEOC Guidelines |
| **2. Explainability** 🔍 | Feature attributions & model complexity | Global SHAP (SHapley Additive exPlanations), Surrogate Decision Tree Fidelity ($R^2$, Accuracy), Proxy Bias | EU AI Act Art. 13 (Transparency), NIST AI RMF "Explainable & Interpretable" |
| **3. Privacy Protection** 🔒 | Sensitive data exposure & memorization | Microsoft Presidio PII scanning (Names, SSNs, Emails, Phones), $k$-Anonymity Equivalence Classes, Membership Inference Attack (MIA) Generalization Gap | GDPR, HIPAA, NIST AI RMF "Privacy-Enhanced" |
| **4. Robustness & Safety** 🛡️ | Prediction stability & adversarial noise | Gaussian Jitter Perturbation ($\sigma=5\%, 10\%, 20\%$), Prediction Flip Rate, Missingness Dropout Resilience, Out-of-Distribution (OOD) 4-Sigma Stress | EU AI Act Art. 15 (Accuracy, Robustness & Cybersecurity) |
| **5. Transparency** 📜 | Documentation completeness & operational boundaries | Automated Model Cards (Mitchell et al.), Dataset Datasheets (Gebru et al.), Governance Compliance Checklist | EU AI Act Technical Documentation (Annex IV), NIST AI RMF GOVERN |

---

## 💻 Headless CLI & CI/CD Pipeline Integration

EthicShield AI includes a command-line interface for automated continuous integration quality gates:

```bash
# Audit custom dataset and model via CLI
python cli/audit_cli.py \
  --dataset real_world_data/german_credit_dataset.csv \
  --model real_world_data/german_credit_model.joblib \
  --output compliance_audit_report.html \
  --json-output audit_artifacts.json
```

---

## 📁 Repository Structure

```
EthicShield-AI/
├── backend/
│   ├── app.py                      # FastAPI server & REST API upload endpoints
│   ├── engine/                     # Multi-pillar diagnostic engines
│   │   ├── auditor.py              # Master orchestrator & composite scoring algorithm
│   │   ├── fairness.py             # Multi-attribute demographic discovery & Fairlearn engine
│   │   ├── explainability.py       # SHAP attributions & surrogate decision trees
│   │   ├── privacy.py              # Microsoft Presidio PII & k-anonymity evaluator
│   │   ├── robustness.py           # Gaussian perturbation jitter & OOD stress tests
│   │   └── transparency.py         # Autonomous Model Card & Datasheet synthesis
│   ├── samples/
│   │   └── generator.py            # Benchmark domain generators (Lending, HR, Clinical)
│   └── reports/
│       └── generator.py            # Standalone HTML & JSON compliance report generators
├── frontend/
│   ├── index.html                  # Enterprise single-page upload portal
│   ├── styles.css                  # Responsive dark enterprise design system
│   └── app.js                      # Client logic & Chart.js multi-demographic visualizations
├── real_world_data/                # Pre-packaged reviewer test artifacts
│   ├── german_credit_dataset.csv   # Real-world German Credit benchmark dataset
│   └── german_credit_model.joblib  # Trained scikit-learn reviewer model
├── cli/
│   └── audit_cli.py                # Headless CI/CD terminal auditing utility
├── tests/
│   └── test_auditor.py             # Comprehensive test suite (All 5 pillars + autonomous test)
├── create_real_world_sample.py     # 1-command generator for reviewer dataset & model
├── requirements.txt                # Production dependency manifest
├── run.py                          # Application entry point (http://localhost:8000)
└── README.md                       # Documentation & reviewer guide
```

---

## 📄 Compliance Standards

* **EU Artificial Intelligence Act (Regulation 2024/1689)**: Specifically addresses High-Risk AI conformity assessments across Articles 9, 10, 11, 13, 14, and 15.
* **NIST AI Risk Management Framework (AI RMF 1.0)**: Operationalizes the GOVERN, MAP, MEASURE, and MANAGE functions across validity, safety, security, fairness, explainability, and privacy.
* **IEEE 7000™ Standard**: Standard model process for addressing ethical concerns during system design.

---

## 📜 License & Citation

Distributed under the Apache-2.0 License. See `LICENSE` for more information.
