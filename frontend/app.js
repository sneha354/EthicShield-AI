/**
 * EthicShield AI - Frontend Application Logic
 * Coordinates benchmark scenario audits, custom uploads, chart rendering, and report exports.
 */

let currentReport = null;
let radarChartInstance = null;
let subgroupChartInstance = null;
let featureImpChartInstance = null;

document.addEventListener("DOMContentLoaded", () => {
    loadScenarios();
});

// Mode switcher: benchmark vs custom
function switchAuditMode(mode) {
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));

    const btnCustom = document.getElementById("tab-btn-custom");
    const modeCustom = document.getElementById("mode-custom");
    if (btnCustom) btnCustom.classList.add("active");
    if (modeCustom) modeCustom.classList.add("active");
}

// Detail tabs switcher
function switchDetailTab(tabId) {
    document.querySelectorAll(".d-tab-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".d-tab-pane").forEach(p => p.classList.remove("active"));

    const btn = Array.from(document.querySelectorAll(".d-tab-btn")).find(b => b.textContent.toLowerCase().includes(tabId));
    if (btn) btn.classList.add("active");

    const pane = document.getElementById(`d-tab-${tabId}`);
    if (pane) pane.classList.add("active");
}

// Fetch benchmark scenarios (if present)
async function loadScenarios() {
    const container = document.getElementById("scenarios-container");
    if (!container) return;
    try {
        const res = await fetch("/api/scenarios");
        const scenarios = await res.json();

        container.innerHTML = scenarios.map(s => `
            <div class="scenario-card">
                <div>
                    <span class="scenario-badge">${s.domain}</span>
                    <h4 class="scenario-title">${s.title}</h4>
                    <p class="scenario-desc">${s.description}</p>
                    <div class="scenario-meta">
                        <div><span>Dataset Records:</span> <strong>${s.rows.toLocaleString()} rows</strong></div>
                        <div><span>Protected Attribute:</span> <strong>${s.sensitive_column}</strong></div>
                        <div><span>Decision Target:</span> <strong>${s.target_column}</strong></div>
                        <div><span>Trained Model:</span> <strong>${s.model_name}</strong></div>
                    </div>
                </div>
                <button class="btn-run-scenario" onclick="runSampleAudit('${s.id}', '${s.title}')">
                    ⚡ Run Comprehensive Audit
                </button>
            </div>
        `).join("");
    } catch (err) {
        container.innerHTML = `<div style="color: #ef4444;">Failed to load benchmark scenarios: ${err.message}</div>`;
    }
}

// Execute 1-click sample audit
async function runSampleAudit(scenarioId, title) {
    showLoading(`Auditing scenario: "${title}" across 5 pillars...`);
    try {
        const res = await fetch(`/api/audit/sample/${scenarioId}`, { method: "POST" });
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || "Audit failed");
        }
        const report = await res.json();
        currentReport = report;
        renderResults(report, title);
    } catch (err) {
        alert(`Audit Error: ${err.message}`);
    } finally {
        hideLoading();
    }
}

// Handle dataset selection & column inspection
async function handleDatasetSelect(event) {
    const file = event.target.files[0];
    if (!file) return;

    document.getElementById("dataset-file-label").textContent = `Selected: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;

    // Inspect columns via API
    const formData = new FormData();
    formData.append("dataset", file);

    try {
        const res = await fetch("/api/inspect-columns", {
            method: "POST",
            body: formData
        });
        if (!res.ok) throw new Error("Could not inspect columns");
        const data = await res.json();

        const sensSelect = document.getElementById("sensitive-col-select");
        const targetSelect = document.getElementById("target-col-select");

        sensSelect.innerHTML = `<option value="auto" selected>✨ Auto-Detect All Demographics (Holistic Audit)</option>` +
            data.columns.map(c => `
                <option value="${c.name}">
                    ${c.name} ${c.is_potential_sensitive ? " (Protected Attr)" : ""}
                </option>
            `).join("");

        targetSelect.innerHTML = `<option value="auto" selected>✨ Auto-Detect Primary Decision Target</option>` +
            data.columns.map(c => `
                <option value="${c.name}">
                    ${c.name} ${c.is_potential_target ? " (Target)" : ""}
                </option>
            `).join("");
    } catch (err) {
        console.error("Column inspection error:", err);
    }
}

function handleModelSelect(event) {
    const file = event.target.files[0];
    if (!file) return;
    document.getElementById("model-file-label").textContent = `Selected: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
}

// Handle custom audit submission
async function handleCustomAudit(event) {
    event.preventDefault();
    const datasetInput = document.getElementById("dataset-file");
    if (!datasetInput.files.length) {
        alert("Please select a dataset file to audit.");
        return;
    }

    const formData = new FormData();
    formData.append("dataset", datasetInput.files[0]);

    const modelInput = document.getElementById("model-file");
    if (modelInput.files.length) {
        formData.append("model_file", modelInput.files[0]);
    }

    const sensCol = document.getElementById("sensitive-col-select") ? document.getElementById("sensitive-col-select").value : "";
    if (sensCol && sensCol !== "auto") formData.append("sensitive_column", sensCol);

    const targetCol = document.getElementById("target-col-select") ? document.getElementById("target-col-select").value : "";
    if (targetCol && targetCol !== "auto") formData.append("target_column", targetCol);

    const privGroup = document.getElementById("privileged-group-input") ? document.getElementById("privileged-group-input").value : "";
    if (privGroup && privGroup.trim() !== "" && privGroup.toLowerCase() !== "auto") {
        formData.append("privileged_group", privGroup.trim());
    }

    const modelName = document.getElementById("model-name-input") ? document.getElementById("model-name-input").value : "";
    if (modelName) formData.append("model_name", modelName);

    const intendedUse = document.getElementById("intended-use-input").value;
    if (intendedUse) formData.append("intended_use", intendedUse);

    const outOfScope = document.getElementById("out-of-scope-input").value;
    if (outOfScope) formData.append("out_of_scope", outOfScope);

    const domain = document.getElementById("domain-input").value;
    if (domain) formData.append("domain", domain);

    const licenseStr = document.getElementById("license-input").value;
    if (licenseStr) formData.append("license_str", licenseStr);

    const humanOversight = document.getElementById("human-oversight-check").checked;
    formData.append("human_oversight", humanOversight);

    showLoading("Running multi-pillar audit on your custom model and dataset...");
    try {
        const res = await fetch("/api/audit/upload", {
            method: "POST",
            body: formData
        });
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || "Audit execution failed");
        }
        const report = await res.json();
        currentReport = report;
        renderResults(report, modelName || datasetInput.files[0].name);
    } catch (err) {
        alert(`Custom Audit Error: ${err.message}`);
    } finally {
        hideLoading();
    }
}

// Render Results Dashboard
function renderResults(report, systemName) {
    const summary = report.summary;
    const scores = report.pillar_scores;
    const pillars = report.pillars;

    // Header info
    document.getElementById("res-system-name").textContent = systemName || "Audited ML System";
    document.getElementById("res-timestamp").textContent = 
        `Audited: ${new Date(summary.audit_timestamp).toLocaleString()} | ${summary.dataset_rows} Instances | Target: ${summary.target_column || "N/A"}`;

    // Overall Score & Badges
    document.getElementById("res-overall-score").textContent = summary.overall_score;
    document.getElementById("res-letter-grade").textContent = `Grade: ${summary.letter_grade}`;

    const riskBadge = document.getElementById("res-risk-badge");
    riskBadge.textContent = summary.status;
    riskBadge.className = "risk-badge " + (
        summary.overall_score >= 80 ? "risk-low" :
        summary.overall_score >= 65 ? "risk-moderate" :
        summary.overall_score >= 50 ? "risk-high" : "risk-critical"
    );

    // Meta items
    document.getElementById("res-rows-count").textContent = `${summary.dataset_rows} rows`;
    document.getElementById("res-cols-count").textContent = `${summary.dataset_columns} columns`;
    document.getElementById("res-sensitive-attr").textContent = summary.sensitive_column || "None";
    document.getElementById("res-model-type").textContent = pillars.explainability.model_type || "N/A";

    // 5 Pillar Cards
    document.getElementById("p-score-fairness").textContent = scores.fairness;
    document.getElementById("p-bar-fairness").style.width = `${scores.fairness}%`;
    const diRatio = pillars.fairness.model_fairness?.disparate_impact_ratio;
    document.getElementById("p-meta-fairness").textContent = diRatio ? `Disparate Impact: ${diRatio}` : "Representation Audited";

    document.getElementById("p-score-explainability").textContent = scores.explainability;
    document.getElementById("p-bar-explainability").style.width = `${scores.explainability}%`;
    document.getElementById("p-meta-explainability").textContent = `Attribution: ${pillars.explainability.attribution_method || "SHAP"}`;

    document.getElementById("p-score-privacy").textContent = scores.privacy;
    document.getElementById("p-bar-privacy").style.width = `${scores.privacy}%`;
    document.getElementById("p-meta-privacy").textContent = `PII: ${pillars.privacy.total_pii_count} instances detected`;

    document.getElementById("p-score-robustness").textContent = scores.robustness;
    document.getElementById("p-bar-robustness").style.width = `${scores.robustness}%`;
    document.getElementById("p-meta-robustness").textContent = `Noise Flip Rate: ${pillars.robustness.prediction_flip_rate}%`;

    document.getElementById("p-score-transparency").textContent = scores.transparency;
    document.getElementById("p-bar-transparency").style.width = `${scores.transparency}%`;
    document.getElementById("p-meta-transparency").textContent = `Doc Completeness: ${pillars.transparency.completeness_score}%`;

    // Prioritized Action Roadmap
    const actionsContainer = document.getElementById("actions-container");
    document.getElementById("actions-count-label").textContent = `${report.prioritized_actions.length} action item(s) identified`;
    actionsContainer.innerHTML = report.prioritized_actions.map(act => {
        const sevClass = act.severity === "CRITICAL" ? "crit" :
                         act.severity === "HIGH" ? "high" :
                         act.severity === "MEDIUM" ? "med" : "low";
        return `
            <div class="action-item ${sevClass}">
                <div class="action-top">
                    <span class="pillar-badge">${act.pillar} &bull; ${act.severity} PRIORITY</span>
                    <span>Pillar Score: ${act.score}/100</span>
                </div>
                <div class="action-msg">${act.recommendation}</div>
                ${act.context ? `<div class="action-ctx">Context: ${act.context}</div>` : ""}
            </div>
        `;
    }).join("");

    // Render Charts
    renderRadarChart(scores);
    renderFairnessDetails(pillars.fairness);
    renderExplainabilityDetails(pillars.explainability);
    renderPrivacyDetails(pillars.privacy);
    renderRobustnessDetails(pillars.robustness);
    renderTransparencyDetails(pillars.transparency);

    // Show results section and scroll smoothly to it
    const resSec = document.getElementById("results-section");
    resSec.classList.remove("hidden");
    resSec.scrollIntoView({ behavior: "smooth" });
}

// Chart: 5 Pillars Radar
function renderRadarChart(scores) {
    const ctx = document.getElementById("ethicsRadarChart").getContext("2d");
    if (radarChartInstance) radarChartInstance.destroy();

    radarChartInstance = new Chart(ctx, {
        type: "radar",
        data: {
            labels: ["Fairness", "Explainability", "Privacy", "Robustness", "Transparency"],
            datasets: [{
                label: "Audit Score",
                data: [scores.fairness, scores.explainability, scores.privacy, scores.robustness, scores.transparency],
                backgroundColor: "rgba(59, 130, 246, 0.25)",
                borderColor: "#3b82f6",
                pointBackgroundColor: "#60a5fa",
                pointBorderColor: "#fff",
                pointHoverBackgroundColor: "#fff",
                pointHoverBorderColor: "#3b82f6"
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    min: 0,
                    max: 100,
                    ticks: { display: false, stepSize: 20 },
                    grid: { color: "#23324d" },
                    angleLines: { color: "#23324d" },
                    pointLabels: { color: "#94a3b8", font: { size: 11, weight: "bold" } }
                }
            },
            plugins: { legend: { display: false } }
        }
    });
}

// Detail Renderers
function renderFairnessDetails(fairness) {
    const mf = fairness.model_fairness || {};
    const tableDiv = document.getElementById("fairness-metrics-table");

    let multiScorecardHtml = "";
    if (fairness.demographic_breakdowns && Object.keys(fairness.demographic_breakdowns).length > 1) {
        multiScorecardHtml = `
            <div style="margin-bottom: 24px; padding: 18px; background: rgba(59, 130, 246, 0.08); border: 1px solid rgba(59, 130, 246, 0.25); border-radius: 8px;">
                <h4 style="margin-bottom: 6px; color: #60a5fa; display: flex; align-items: center; gap: 8px;">
                    <span>🌐</span> Multi-Demographic Fairness Scorecard (Autonomous Discovery)
                </h4>
                <p style="font-size: 13px; color: #94a3b8; margin-bottom: 14px;">
                    EthicShield AI discovered <strong>${Object.keys(fairness.demographic_breakdowns).length} protected demographic dimensions</strong> in this dataset and evaluated algorithmic parity across each dimension independently:
                </p>
                <table class="app-table">
                    <thead>
                        <tr>
                            <th>Demographic Attribute</th>
                            <th>Groups Analyzed</th>
                            <th>Privileged Baseline</th>
                            <th>Disparate Impact (80% Rule)</th>
                            <th>Pillar Score</th>
                            <th>Compliance Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${Object.entries(fairness.demographic_breakdowns).map(([attr, data]) => {
                            const di = (data.model_fairness && data.model_fairness.disparate_impact_ratio !== undefined) ? data.model_fairness.disparate_impact_ratio : "N/A";
                            const pass = data.model_fairness ? data.model_fairness.four_fifths_rule_passed : true;
                            const baseline = data.privileged_group || "Inferred Baseline";
                            const grps = data.groups ? data.groups.join(", ") : "All";
                            return `
                                <tr>
                                    <td><strong>${attr}</strong></td>
                                    <td><span style="font-size: 12px; color: #cbd5e1;">${grps}</span></td>
                                    <td><span class="badge" style="background:#1e293b; color:#93c5fd; padding:3px 8px; font-size:11px;">${baseline}</span></td>
                                    <td><strong>${di}</strong></td>
                                    <td><strong>${data.score}/100</strong></td>
                                    <td><strong style="color: ${pass ? '#10b981' : '#ef4444'}">${pass ? 'PASS (&ge; 0.80)' : 'VIOLATION (< 0.80)'}</strong></td>
                                </tr>
                            `;
                        }).join("")}
                    </tbody>
                </table>
            </div>
            <h4 style="margin-bottom: 12px; color: #f8fafc;">Primary Demographic Focus: <code>${fairness.sensitive_column}</code></h4>
        `;
    }

    tableDiv.innerHTML = multiScorecardHtml + `
        <table class="app-table">
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Observed</th>
                    <th>Threshold</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>Disparate Impact Ratio</strong></td>
                    <td>${mf.disparate_impact_ratio !== undefined ? mf.disparate_impact_ratio : "N/A"}</td>
                    <td>&ge; 0.80 (80% Rule)</td>
                    <td><strong style="color: ${mf.four_fifths_rule_passed ? '#10b981' : '#ef4444'}">
                        ${mf.four_fifths_rule_passed ? 'PASS' : 'VIOLATION'}
                    </strong></td>
                </tr>
                <tr>
                    <td><strong>Demographic Parity Gap</strong></td>
                    <td>${mf.demographic_parity_difference !== undefined ? mf.demographic_parity_difference : "N/A"}</td>
                    <td>&le; 0.10</td>
                    <td><strong style="color: ${mf.demographic_parity_difference <= 0.10 ? '#10b981' : '#f59e0b'}">
                        ${mf.demographic_parity_difference <= 0.10 ? 'PASS' : 'WARNING'}
                    </strong></td>
                </tr>
                ${mf.equal_opportunity_difference !== undefined ? `
                <tr>
                    <td><strong>Equal Opportunity Gap</strong></td>
                    <td>${mf.equal_opportunity_difference}</td>
                    <td>&le; 0.10</td>
                    <td><strong style="color: ${mf.equal_opportunity_difference <= 0.10 ? '#10b981' : '#f59e0b'}">
                        ${mf.equal_opportunity_difference <= 0.10 ? 'PASS' : 'WARNING'}
                    </strong></td>
                </tr>
                ` : ""}
            </tbody>
        </table>
    `;

    // Subgroup breakdown table & chart
    const subTable = document.getElementById("subgroup-breakdown-table");
    const subgroups = fairness.subgroup_metrics || [];

    if (subgroups.length > 0) {
        subTable.innerHTML = `
            <table class="app-table">
                <thead>
                    <tr>
                        <th>Demographic Group</th>
                        <th>Sample Size</th>
                        <th>Selection Rate (Favorable Outcome)</th>
                        <th>Accuracy</th>
                        <th>True Positive Rate (Recall)</th>
                        <th>False Positive Rate</th>
                    </tr>
                </thead>
                <tbody>
                    ${subgroups.map(g => `
                        <tr>
                            <td><strong>${g.group}</strong></td>
                            <td>${g.count}</td>
                            <td>${(g.selection_rate * 100).toFixed(1)}%</td>
                            <td>${g.accuracy !== undefined ? (g.accuracy * 100).toFixed(1) + "%" : "N/A"}</td>
                            <td>${g.true_positive_rate !== null && g.true_positive_rate !== undefined ? (g.true_positive_rate * 100).toFixed(1) + "%" : "N/A"}</td>
                            <td>${g.false_positive_rate !== null && g.false_positive_rate !== undefined ? (g.false_positive_rate * 100).toFixed(1) + "%" : "N/A"}</td>
                        </tr>
                    `).join("")}
                </tbody>
            </table>
        `;

        // Bar chart for selection rates
        const ctx = document.getElementById("subgroupChart").getContext("2d");
        if (subgroupChartInstance) subgroupChartInstance.destroy();

        subgroupChartInstance = new Chart(ctx, {
            type: "bar",
            data: {
                labels: subgroups.map(g => g.group),
                datasets: [{
                    label: "Selection Rate %",
                    data: subgroups.map(g => (g.selection_rate * 100).toFixed(1)),
                    backgroundColor: "#38bdf8"
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { min: 0, max: 100, grid: { color: "#23324d" } },
                    x: { grid: { display: false } }
                }
            }
        });
    } else {
        subTable.innerHTML = `<p style="color: #94a3b8; font-size: 13px;">No model subgroup predictions available.</p>`;
    }
}

function renderExplainabilityDetails(exp) {
    const kpiDiv = document.getElementById("explainability-kpi-list");
    kpiDiv.innerHTML = `
        <div class="kpi-row">
            <span>Model Family:</span>
            <strong>${exp.model_family || exp.model_type}</strong>
        </div>
        <div class="kpi-row">
            <span>Inherent Interpretability:</span>
            <strong>${exp.complexity_level}</strong>
        </div>
        <div class="kpi-row">
            <span>Attribution Method:</span>
            <strong>${exp.attribution_method}</strong>
        </div>
        ${exp.surrogate_fidelity ? `
        <div class="kpi-row">
            <span>Surrogate Decision Tree Fidelity:</span>
            <strong>${(exp.surrogate_fidelity.fidelity_score * 100).toFixed(1)}% (Depth ${exp.surrogate_fidelity.interpretable_depth})</strong>
        </div>
        ` : ""}
        ${exp.feature_concentration ? `
        <div class="kpi-row">
            <span>Top Feature Share:</span>
            <strong>${exp.feature_concentration.top_feature} (${exp.feature_concentration.top_1_percentage}%)</strong>
        </div>
        ` : ""}
    `;

    // Feature importance chart
    const feats = exp.feature_attributions || [];
    if (feats.length > 0) {
        const ctx = document.getElementById("featureImpChart").getContext("2d");
        if (featureImpChartInstance) featureImpChartInstance.destroy();

        const topFeats = feats.slice(0, 8);
        featureImpChartInstance = new Chart(ctx, {
            type: "bar",
            data: {
                labels: topFeats.map(f => f.feature),
                datasets: [{
                    label: "% Decision Influence",
                    data: topFeats.map(f => f.percentage),
                    backgroundColor: "#a855f7"
                }]
            },
            options: {
                indexAxis: "y",
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { min: 0, grid: { color: "#23324d" } },
                    y: { grid: { display: false } }
                }
            }
        });
    }
}

function renderPrivacyDetails(priv) {
    const tableDiv = document.getElementById("pii-findings-table");
    const piiList = priv.pii_detected || [];

    if (piiList.length > 0) {
        tableDiv.innerHTML = `
            <table class="app-table">
                <thead>
                    <tr>
                        <th>Column</th>
                        <th>PII Entity Type</th>
                        <th>Occurrences</th>
                        <th>Confidence</th>
                        <th>Masked Preview</th>
                    </tr>
                </thead>
                <tbody>
                    ${piiList.map(p => `
                        <tr>
                            <td><strong>${p.column}</strong></td>
                            <td><span style="color: #f87171; font-weight: 700;">${p.entity_type}</span></td>
                            <td>${p.count}</td>
                            <td>${p.confidence}</td>
                            <td><code>${p.sample}</code></td>
                        </tr>
                    `).join("")}
                </tbody>
            </table>
        `;
    } else {
        tableDiv.innerHTML = `<p style="color: #34d399; font-size: 13px; padding: 12px 0;">🛡️ No direct PII entities detected in audited sample.</p>`;
    }

    const kpiDiv = document.getElementById("privacy-kpi-list");
    const kAnon = priv.k_anonymity || {};
    const mia = priv.membership_inference_risk || {};

    kpiDiv.innerHTML = `
        <div class="kpi-row">
            <span>Total PII Instances Detected:</span>
            <strong style="color: ${priv.total_pii_count > 0 ? '#f87171' : '#34d399'}">${priv.total_pii_count}</strong>
        </div>
        ${kAnon.min_k !== undefined ? `
        <div class="kpi-row">
            <span>k-Anonymity (Quasi-Identifiers):</span>
            <strong>Min k = ${kAnon.min_k} (${kAnon.unique_records_pct}% unique records)</strong>
        </div>
        ` : ""}
        ${mia.generalization_gap !== undefined ? `
        <div class="kpi-row">
            <span>Membership Inference Vulnerability:</span>
            <strong>${(mia.generalization_gap * 100).toFixed(1)}% Gen Gap (${mia.risk_assessment} Risk)</strong>
        </div>
        ` : ""}
    `;
}

function renderRobustnessDetails(rob) {
    const tableDiv = document.getElementById("noise-stress-table");
    const tests = rob.noise_perturbation_tests || [];

    if (tests.length > 0) {
        tableDiv.innerHTML = `
            <table class="app-table">
                <thead>
                    <tr>
                        <th>Gaussian Jitter Magnitude</th>
                        <th>Flipped Predictions</th>
                        <th>Flip Rate %</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    ${tests.map(t => `
                        <tr>
                            <td><strong>${t.noise_magnitude}</strong></td>
                            <td>${t.flipped_predictions_count}</td>
                            <td>${t.flip_rate_pct}%</td>
                            <td><strong style="color: ${t.status === 'PASS' ? '#34d399' : t.status === 'WARNING' ? '#fbbf24' : '#f87171'}">${t.status}</strong></td>
                        </tr>
                    `).join("")}
                </tbody>
            </table>
        `;
    } else {
        tableDiv.innerHTML = `<p style="color: #94a3b8; font-size: 13px;">Upload a model to evaluate noise perturbation stress tests.</p>`;
    }

    const kpiDiv = document.getElementById("robustness-kpi-list");
    const da = rob.dataset_anomalies || {};
    const miss = rob.missing_value_resilience || {};
    const ood = rob.ood_stability || {};

    kpiDiv.innerHTML = `
        <div class="kpi-row">
            <span>Dataset Outliers Ratio:</span>
            <strong>${da.outlier_pct || 0}%</strong>
        </div>
        <div class="kpi-row">
            <span>Missing Data Rate:</span>
            <strong>${da.missing_data_pct || 0}%</strong>
        </div>
        <div class="kpi-row">
            <span>Duplicate Records:</span>
            <strong>${da.duplicate_pct || 0}%</strong>
        </div>
        ${miss.resilience_status ? `
        <div class="kpi-row">
            <span>Missing Value Resilience (10% Dropout):</span>
            <strong>${miss.resilience_status} Resilience (${miss.prediction_drift_pct}% drift)</strong>
        </div>
        ` : ""}
        ${ood.mean_prediction_confidence ? `
        <div class="kpi-row">
            <span>OOD 4-Sigma Boundary Confidence:</span>
            <strong>${(ood.mean_prediction_confidence * 100).toFixed(1)}% (${ood.overconfidence_risk} Overconfidence)</strong>
        </div>
        ` : ""}
    `;
}

function renderTransparencyDetails(trans) {
    const tableDiv = document.getElementById("governance-checklist-table");
    const checklist = trans.governance_checklist || [];

    tableDiv.innerHTML = `
        <table class="app-table">
            <thead>
                <tr>
                    <th>Governance Dimension</th>
                    <th>Weight</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                ${checklist.map(c => `
                    <tr>
                        <td><strong>${c.title}</strong><br><small style="color: #94a3b8">${c.description}</small></td>
                        <td>${c.weight}%</td>
                        <td><strong style="color: ${c.status === 'PASSED' ? '#34d399' : '#f87171'}">${c.status}</strong></td>
                    </tr>
                `).join("")}
            </tbody>
        </table>
    `;

    const mc = trans.model_card || {};
    const cardDiv = document.getElementById("model-card-preview");
    cardDiv.innerHTML = `
        <h5>Model Details</h5>
        <p><strong>Name:</strong> ${mc.model_name || "N/A"} (v${mc.version || "1.0"})</p>
        <p><strong>Created:</strong> ${mc.creation_date || "N/A"}</p>
        <h5>Intended Use</h5>
        <p>${mc.intended_use || "N/A"}</p>
        <h5>Out-of-Scope / Prohibited Uses</h5>
        <p>${mc.out_of_scope_uses || "N/A"}</p>
        <h5>Ethical Considerations</h5>
        <p>${mc.ethical_considerations || "N/A"}</p>
    `;
}

// Export HTML or JSON reports
async function exportReport(format) {
    if (!currentReport) {
        alert("Please run an audit before exporting.");
        return;
    }

    try {
        const endpoint = format === "html" ? "/api/export/html" : "/api/export/json";
        const res = await fetch(endpoint, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(currentReport)
        });

        if (!res.ok) throw new Error("Export request failed");

        const blob = await res.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = format === "html" ? "ai_ethics_audit_report.html" : "ai_ethics_audit_report.json";
        document.body.appendChild(a);
        a.click();
        a.remove();
        window.URL.revokeObjectURL(url);
    } catch (err) {
        alert(`Export failed: ${err.message}`);
    }
}

// Loading UI helpers
function showLoading(msg) {
    const p = document.getElementById("audit-progress");
    p.classList.remove("hidden");
    if (msg) document.getElementById("progress-steps-label").textContent = msg;
    p.scrollIntoView({ behavior: "smooth" });
}

function hideLoading() {
    document.getElementById("audit-progress").classList.add("hidden");
}

