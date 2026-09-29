"""
HTML & JSON Report Generator for AI Ethics Auditor
Generates clean, printable, standalone audit compliance documents and JSON export artifacts.
"""

from typing import Dict, Any
import json
from jinja2 import Template


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Ethics & Safety Audit Report - {{ summary.model_name or 'ML System' }}</title>
    <style>
        :root {
            --primary: #2563eb;
            --primary-dark: #1d4ed8;
            --success: #16a34a;
            --warning: #d97706;
            --danger: #dc2626;
            --dark: #0f172a;
            --card-bg: #ffffff;
            --border: #e2e8f0;
            --text: #1e293b;
            --text-muted: #64748b;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            color: var(--text);
            background: #f8fafc;
            line-height: 1.6;
            padding: 40px 20px;
        }
        .container {
            max-width: 1000px;
            margin: 0 auto;
            background: var(--card-bg);
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.06);
            padding: 48px;
            border: 1px solid var(--border);
        }
        .header {
            border-bottom: 2px solid var(--border);
            padding-bottom: 24px;
            margin-bottom: 32px;
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
        }
        .header-title h1 {
            font-size: 28px;
            font-weight: 800;
            color: var(--dark);
            letter-spacing: -0.5px;
        }
        .header-title p {
            color: var(--text-muted);
            font-size: 14px;
            margin-top: 6px;
        }
        .badge {
            display: inline-block;
            padding: 6px 14px;
            border-radius: 9999px;
            font-size: 13px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .badge-low { background: #dcfce7; color: #15803d; }
        .badge-moderate { background: #fef3c7; color: #b45309; }
        .badge-high { background: #fee2e2; color: #b91c1c; }
        .badge-critical { background: #7f1d1d; color: #ffffff; }

        /* Executive Score Banner */
        .executive-banner {
            display: grid;
            grid-template-columns: 220px 1fr;
            gap: 28px;
            background: #f1f5f9;
            border-radius: 10px;
            padding: 28px;
            margin-bottom: 36px;
            align-items: center;
        }
        .score-circle {
            text-align: center;
            background: #ffffff;
            border-radius: 12px;
            padding: 20px;
            border: 1px solid var(--border);
        }
        .score-num {
            font-size: 54px;
            font-weight: 900;
            line-height: 1;
            color: var(--primary);
        }
        .score-grade {
            font-size: 20px;
            font-weight: 700;
            color: var(--text-muted);
            margin-top: 4px;
        }
        .score-meta h3 {
            font-size: 20px;
            font-weight: 700;
            margin-bottom: 8px;
            color: var(--dark);
        }

        /* Pillars Grid */
        .pillars-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
            gap: 16px;
            margin-bottom: 36px;
        }
        .pillar-card {
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px;
            background: #ffffff;
            text-align: center;
        }
        .pillar-card h4 {
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-muted);
            margin-bottom: 8px;
        }
        .pillar-score {
            font-size: 26px;
            font-weight: 800;
            color: var(--dark);
        }
        .pillar-bar {
            height: 6px;
            background: #e2e8f0;
            border-radius: 3px;
            margin-top: 10px;
            overflow: hidden;
        }
        .pillar-bar-fill {
            height: 100%;
            background: var(--primary);
            border-radius: 3px;
        }

        /* Section styles */
        .section {
            margin-bottom: 40px;
        }
        .section-title {
            font-size: 20px;
            font-weight: 700;
            color: var(--dark);
            border-bottom: 1px solid var(--border);
            padding-bottom: 8px;
            margin-bottom: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        /* Tables */
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 12px;
            font-size: 14px;
        }
        th, td {
            padding: 12px 14px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }
        th {
            background: #f8fafc;
            color: var(--text-muted);
            font-weight: 600;
            text-transform: uppercase;
            font-size: 12px;
            letter-spacing: 0.5px;
        }

        /* Action items list */
        .action-card {
            border-left: 4px solid var(--border);
            padding: 14px 18px;
            margin-bottom: 12px;
            background: #f8fafc;
            border-radius: 0 8px 8px 0;
        }
        .action-critical { border-left-color: var(--danger); background: #fef2f2; }
        .action-high { border-left-color: #ea580c; background: #fff7ed; }
        .action-medium { border-left-color: var(--warning); background: #fffbeb; }
        .action-low { border-left-color: var(--success); background: #f0fdf4; }

        .action-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 4px;
        }
        .action-pillar {
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .action-text {
            font-size: 14px;
            color: var(--text);
            font-weight: 500;
        }
        .action-context {
            font-size: 12px;
            color: var(--text-muted);
            margin-top: 4px;
        }

        /* Footer */
        .footer {
            margin-top: 48px;
            padding-top: 20px;
            border-top: 1px solid var(--border);
            font-size: 12px;
            color: var(--text-muted);
            text-align: center;
        }
        @media print {
            body { background: white; padding: 0; }
            .container { box-shadow: none; border: none; padding: 20px; }
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <div class="header-title">
                <h1>AI Ethics & Safety Audit Report</h1>
                <p>Audited: {{ summary.audit_timestamp }} | Sensitive Attribute: {{ summary.sensitive_column or 'None' }} | Target: {{ summary.target_column or 'None' }}</p>
            </div>
            <div>
                <span class="badge badge-{{ pillars.fairness.risk_level.lower() }}">{{ summary.status }}</span>
            </div>
        </div>

        <!-- Executive Banner -->
        <div class="executive-banner">
            <div class="score-circle">
                <div class="score-num">{{ summary.overall_score }}</div>
                <div class="score-grade">Grade: {{ summary.letter_grade }}</div>
            </div>
            <div class="score-meta">
                <h3>Overall Ethics Assessment</h3>
                <p>This ML system scored <strong>{{ summary.overall_score }}/100</strong> across Explainability, Transparency, Fairness, Privacy, and Robustness. Evaluated on {{ summary.dataset_rows }} instances with {{ summary.dataset_columns }} features.</p>
            </div>
        </div>

        <!-- 5 Pillars Scores -->
        <div class="pillars-grid">
            <div class="pillar-card">
                <h4>Fairness</h4>
                <div class="pillar-score">{{ pillar_scores.fairness }}</div>
                <div class="pillar-bar"><div class="pillar-bar-fill" style="width: {{ pillar_scores.fairness }}%"></div></div>
            </div>
            <div class="pillar-card">
                <h4>Explainability</h4>
                <div class="pillar-score">{{ pillar_scores.explainability }}</div>
                <div class="pillar-bar"><div class="pillar-bar-fill" style="width: {{ pillar_scores.explainability }}%"></div></div>
            </div>
            <div class="pillar-card">
                <h4>Privacy</h4>
                <div class="pillar-score">{{ pillar_scores.privacy }}</div>
                <div class="pillar-bar"><div class="pillar-bar-fill" style="width: {{ pillar_scores.privacy }}%"></div></div>
            </div>
            <div class="pillar-card">
                <h4>Robustness</h4>
                <div class="pillar-score">{{ pillar_scores.robustness }}</div>
                <div class="pillar-bar"><div class="pillar-bar-fill" style="width: {{ pillar_scores.robustness }}%"></div></div>
            </div>
            <div class="pillar-card">
                <h4>Transparency</h4>
                <div class="pillar-score">{{ pillar_scores.transparency }}</div>
                <div class="pillar-bar"><div class="pillar-bar-fill" style="width: {{ pillar_scores.transparency }}%"></div></div>
            </div>
        </div>

        <!-- Prioritized Action Roadmap -->
        <div class="section">
            <div class="section-title">
                <span>Prioritized Remediation Roadmap</span>
                <span style="font-size: 13px; font-weight: normal; color: var(--text-muted);">{{ prioritized_actions | length }} items</span>
            </div>
            {% for action in prioritized_actions %}
            <div class="action-card action-{{ action.severity.lower() }}">
                <div class="action-header">
                    <span class="action-pillar">{{ action.pillar }} ({{ action.severity }})</span>
                    <span style="font-size: 12px; font-weight: 600;">Score: {{ action.score }}/100</span>
                </div>
                <div class="action-text">{{ action.recommendation }}</div>
                {% if action.context %}
                <div class="action-context">Context: {{ action.context }}</div>
                {% endif %}
            </div>
            {% endfor %}
        </div>

        <!-- Fairness Pillar Details -->
        <div class="section">
            <div class="section-title">
                <span>1. Fairness & Bias Assessment</span>
                <span class="badge badge-{{ pillars.fairness.risk_level.lower() }}">{{ pillars.fairness.score }}/100</span>
            </div>
            {% if pillars.fairness.model_fairness %}
            <table>
                <thead>
                    <tr>
                        <th>Fairness Metric</th>
                        <th>Observed Value</th>
                        <th>Regulatory Standard</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Disparate Impact Ratio</strong></td>
                        <td>{{ pillars.fairness.model_fairness.disparate_impact_ratio }}</td>
                        <td>&ge; 0.80 (Four-Fifths Rule)</td>
                        <td>{{ 'PASS' if pillars.fairness.model_fairness.four_fifths_rule_passed else 'VIOLATION' }}</td>
                    </tr>
                    <tr>
                        <td><strong>Demographic Parity Difference</strong></td>
                        <td>{{ pillars.fairness.model_fairness.demographic_parity_difference }}</td>
                        <td>&le; 0.10 (Parity gap)</td>
                        <td>{{ 'PASS' if pillars.fairness.model_fairness.demographic_parity_difference <= 0.10 else 'WARNING' }}</td>
                    </tr>
                    {% if pillars.fairness.model_fairness.equal_opportunity_difference is defined %}
                    <tr>
                        <td><strong>Equal Opportunity Difference</strong></td>
                        <td>{{ pillars.fairness.model_fairness.equal_opportunity_difference }}</td>
                        <td>&le; 0.10 (TPR disparity)</td>
                        <td>{{ 'PASS' if pillars.fairness.model_fairness.equal_opportunity_difference <= 0.10 else 'WARNING' }}</td>
                    </tr>
                    {% endif %}
                </tbody>
            </table>
            {% endif %}

            {% if pillars.fairness.subgroup_metrics %}
            <h4 style="margin-top: 20px; font-size: 15px; color: var(--dark);">Subgroup Performance Breakdown</h4>
            <table>
                <thead>
                    <tr>
                        <th>Group</th>
                        <th>Sample Size</th>
                        <th>Selection Rate</th>
                        {% if pillars.fairness.subgroup_metrics[0].accuracy is defined %}
                        <th>Accuracy</th>
                        <th>True Pos Rate</th>
                        <th>False Pos Rate</th>
                        {% endif %}
                    </tr>
                </thead>
                <tbody>
                    {% for g in pillars.fairness.subgroup_metrics %}
                    <tr>
                        <td><strong>{{ g.group }}</strong></td>
                        <td>{{ g.count }}</td>
                        <td>{{ (g.selection_rate * 100) | round(1) }}%</td>
                        {% if g.accuracy is defined %}
                        <td>{{ (g.accuracy * 100) | round(1) }}%</td>
                        <td>{{ (g.true_positive_rate * 100) | round(1) if g.true_positive_rate is not none else 'N/A' }}%</td>
                        <td>{{ (g.false_positive_rate * 100) | round(1) if g.false_positive_rate is not none else 'N/A' }}%</td>
                        {% endif %}
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            {% endif %}
        </div>

        <!-- Explainability Pillar Details -->
        <div class="section">
            <div class="section-title">
                <span>2. Explainability & Interpretability</span>
                <span class="badge badge-{{ pillars.explainability.risk_level.lower() }}">{{ pillars.explainability.score }}/100</span>
            </div>
            <p style="font-size: 14px; margin-bottom: 12px;"><strong>Method:</strong> {{ pillars.explainability.attribution_method }} | <strong>Model Type:</strong> {{ pillars.explainability.model_type }} ({{ pillars.explainability.complexity_level }} Interpretability)</p>
            {% if pillars.explainability.feature_attributions %}
            <table>
                <thead>
                    <tr>
                        <th>Top Features</th>
                        <th>Importance Value</th>
                        <th>Share of Decisions</th>
                    </tr>
                </thead>
                <tbody>
                    {% for feat in pillars.explainability.feature_attributions[:6] %}
                    <tr>
                        <td><strong>{{ feat.feature }}</strong></td>
                        <td>{{ feat.importance }}</td>
                        <td>{{ feat.percentage }}%</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            {% endif %}
        </div>

        <!-- Privacy Pillar Details -->
        <div class="section">
            <div class="section-title">
                <span>3. Privacy & Data Protection</span>
                <span class="badge badge-{{ pillars.privacy.risk_level.lower() }}">{{ pillars.privacy.score }}/100</span>
            </div>
            <p style="font-size: 14px;"><strong>Total PII Detected:</strong> {{ pillars.privacy.total_pii_count }} instances</p>
            {% if pillars.privacy.pii_detected %}
            <table>
                <thead>
                    <tr>
                        <th>Column</th>
                        <th>PII Entity Type</th>
                        <th>Occurrences</th>
                        <th>Sample Preview (Masked)</th>
                    </tr>
                </thead>
                <tbody>
                    {% for pii in pillars.privacy.pii_detected %}
                    <tr>
                        <td><strong>{{ pii.column }}</strong></td>
                        <td><span style="color: var(--danger); font-weight: 600;">{{ pii.entity_type }}</span></td>
                        <td>{{ pii.count }}</td>
                        <td><code>{{ pii.sample }}</code></td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            {% endif %}
        </div>

        <!-- Robustness Pillar Details -->
        <div class="section">
            <div class="section-title">
                <span>4. Robustness & Safety Stress Tests</span>
                <span class="badge badge-{{ pillars.robustness.risk_level.lower() }}">{{ pillars.robustness.score }}/100</span>
            </div>
            <p style="font-size: 14px; margin-bottom: 10px;"><strong>Noise Perturbation Flip Rate (5% Jitter):</strong> {{ pillars.robustness.prediction_flip_rate }}% predictions flipped</p>
            {% if pillars.robustness.noise_perturbation_tests %}
            <table>
                <thead>
                    <tr>
                        <th>Noise Magnitude</th>
                        <th>Flipped Predictions</th>
                        <th>Flip Rate %</th>
                        <th>Test Status</th>
                    </tr>
                </thead>
                <tbody>
                    {% for test in pillars.robustness.noise_perturbation_tests %}
                    <tr>
                        <td>{{ test.noise_magnitude }}</td>
                        <td>{{ test.flipped_predictions_count }}</td>
                        <td>{{ test.flip_rate_pct }}%</td>
                        <td><strong>{{ test.status }}</strong></td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            {% endif %}
        </div>

        <!-- Transparency Pillar Details -->
        <div class="section">
            <div class="section-title">
                <span>5. Transparency & Model Card</span>
                <span class="badge badge-{{ pillars.transparency.risk_level.lower() }}">{{ pillars.transparency.score }}/100</span>
            </div>
            <table>
                <thead>
                    <tr>
                        <th>Governance Dimension</th>
                        <th>Description</th>
                        <th>Weight</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {% for check in pillars.transparency.governance_checklist %}
                    <tr>
                        <td><strong>{{ check.title }}</strong></td>
                        <td>{{ check.description }}</td>
                        <td>{{ check.weight }}%</td>
                        <td><strong>{{ check.status }}</strong></td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>

        <!-- Footer -->
        <div class="footer">
            <p>Generated by <strong>EthicShield AI - AI Ethics & Safety Auditor</strong>. Complies with EU AI Act, NIST AI RMF, and IEEE 7000 standards.</p>
        </div>
    </div>
</body>
</html>
"""


def generate_html_report(report_data: Dict[str, Any]) -> str:
    """Renders self-contained HTML audit document from report dictionary."""
    template = Template(HTML_TEMPLATE)
    return template.render(**report_data)


def export_json_report(report_data: Dict[str, Any], indent: int = 2) -> str:
    """Serializes audit report to structured JSON string."""
    def json_serial(obj):
        if hasattr(obj, "item"):
            return obj.item()
        if hasattr(obj, "tolist"):
            return obj.tolist()
        return str(obj)

    return json.dumps(report_data, indent=indent, default=json_serial)

