"""
FastAPI Server for AI Ethics Auditor (EthicShield AI)
Provides REST endpoints for auditing datasets & models, exploring benchmark scenarios,
and exporting compliance documents.
"""

import os
import sys
import io
import json
import joblib
import pandas as pd
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Ensure root directory is on sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.engine.auditor import AIEthicsAuditor
from backend.samples.generator import get_scenario, list_scenarios
from backend.reports.generator import generate_html_report, export_json_report

app = FastAPI(
    title="EthicShield AI - AI Ethics & Safety Auditor",
    description="Multi-pillar auditing platform for Explainability, Transparency, Fairness, Privacy, and Robustness.",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

auditor = AIEthicsAuditor()


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "app": "EthicShield AI Ethics Auditor", "version": "1.0.0"}


@app.get("/api/scenarios")
async def get_benchmark_scenarios():
    """Returns list of pre-configured audit scenarios."""
    return list_scenarios()


@app.post("/api/audit/sample/{scenario_id}")
async def audit_sample_scenario(scenario_id: str):
    """Execute complete ethics audit on a pre-configured scenario."""
    try:
        df, model, meta = get_scenario(scenario_id)
        report = auditor.audit(
            df=df,
            sensitive_column=meta.get("sensitive_column"),
            target_column=meta.get("target_column"),
            feature_columns=meta.get("feature_columns"),
            model=model,
            privileged_group=meta.get("privileged_group"),
            train_acc=meta.get("train_acc"),
            test_acc=meta.get("test_acc"),
            metadata=meta
        )
        return report
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audit execution error: {str(e)}")


@app.post("/api/audit/upload")
async def audit_uploaded_artifacts(
    dataset: UploadFile = File(...),
    model_file: Optional[UploadFile] = File(None),
    sensitive_column: Optional[str] = Form(None),
    target_column: Optional[str] = Form(None),
    privileged_group: Optional[str] = Form(None),
    model_name: Optional[str] = Form(None),
    intended_use: Optional[str] = Form(None),
    out_of_scope: Optional[str] = Form(None),
    license_str: Optional[str] = Form(None),
    domain: Optional[str] = Form(None),
    human_oversight: Optional[bool] = Form(True)
):
    """
    Audit user-uploaded dataset and optional ML model.
    """
    try:
        # Read dataset
        contents = await dataset.read()
        filename = dataset.filename.lower()

        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        elif filename.endswith(".json"):
            df = pd.read_json(io.BytesIO(contents))
        elif filename.endswith(".parquet"):
            df = pd.read_parquet(io.BytesIO(contents))
        else:
            # Fallback attempt CSV
            df = pd.read_csv(io.BytesIO(contents))

        # Read optional model
        model = None
        if model_file and model_file.filename:
            model_bytes = await model_file.read()
            if len(model_bytes) > 0:
                try:
                    model = joblib.load(io.BytesIO(model_bytes))
                except Exception as me:
                    print(f"Warning: could not deserialize model: {me}")

        metadata = {
            "model_name": model_name or (model.__class__.__name__ if model else "Custom Classifier"),
            "dataset_name": dataset.filename,
            "intended_use": intended_use or "Decision support in target operational domain",
            "out_of_scope": out_of_scope or "Autonomous mission-critical decisions without human verification",
            "license": license_str or "Not Specified",
            "domain": domain or "General ML Application",
            "human_oversight": human_oversight
        }

        # Clean auto parameters for full autonomous execution
        sens_param = None if (not sensitive_column or sensitive_column.strip().lower() == "auto") else sensitive_column.strip()
        target_param = None if (not target_column or target_column.strip().lower() == "auto") else target_column.strip()
        priv_param = None if (not privileged_group or privileged_group.strip().lower() == "auto") else privileged_group.strip()

        report = auditor.audit(
            df=df,
            sensitive_column=sens_param,
            target_column=target_param,
            privileged_group=priv_param,
            model=model,
            metadata=metadata
        )
        return report

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to audit uploaded data: {str(e)}")


@app.post("/api/inspect-columns")
async def inspect_columns(dataset: UploadFile = File(...)):
    """Quick preview of columns, data types, and potential protected attributes."""
    try:
        contents = await dataset.read()
        filename = dataset.filename.lower()
        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents), nrows=20)
        elif filename.endswith(".json"):
            df = pd.read_json(io.BytesIO(contents)).head(20)
        else:
            df = pd.read_csv(io.BytesIO(contents), nrows=20)

        cols_info = []
        for col in df.columns:
            unique_vals = df[col].dropna().unique()[:5].tolist()
            # convert any non-serializable objects
            sample_preview = [str(x) for x in unique_vals]
            cols_info.append({
                "name": col,
                "type": str(df[col].dtype),
                "is_numeric": bool(pd.api.types.is_numeric_dtype(df[col])),
                "sample_values": sample_preview,
                "is_potential_sensitive": any(term in col.lower() for term in ["gender", "sex", "age", "race", "ethnicity", "religion", "disability"]),
                "is_potential_target": any(term in col.lower() for term in ["target", "label", "approved", "class", "outcome", "y", "admitted", "readmitted", "callback"])
            })

        return {"filename": dataset.filename, "columns": cols_info}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Column inspection error: {str(e)}")


@app.post("/api/export/html")
async def export_html(report_data: dict):
    """Generate standalone HTML report download."""
    try:
        html = generate_html_report(report_data)
        return HTMLResponse(
            content=html,
            headers={"Content-Disposition": 'attachment; filename="ai_ethics_audit_report.html"'}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"HTML export failed: {str(e)}")


@app.post("/api/export/json")
async def export_json(report_data: dict):
    """Export machine-readable JSON artifact download."""
    try:
        json_content = export_json_report(report_data)
        return Response(
            content=json_content,
            media_type="application/json",
            headers={"Content-Disposition": 'attachment; filename="ai_ethics_audit_report.json"'}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"JSON export failed: {str(e)}")


# Mount frontend static files
frontend_dir = os.path.join(BASE_DIR, "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

