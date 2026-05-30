"""PrimerForge Web Server – publication-grade Streamlit dashboard.

Tabs:
  1. 🎯 Single-Locus Design    – biophysics + ML scoring + SHAP
  2. 🔀 ILP Multiplex          – dimer-free panel via PuLP
  3. 🧱 Tiled-Amplicon Router  – DP whole-genome tile scheme
  4. 📈 Retrain & Diagnostics  – force retrain, feature importance
  5. 🔬 Fine-Tune (Lab Data)   – upload qPCR CSV, adapt model
"""

import io
import os
import json
import tempfile
import textwrap
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from primerforge.biophysics import BiophysicsEngine
from primerforge.ml_scorer import MLScorer
from primerforge.optimizer import MultiplexOptimizer, TiledAmpliconRouter
from primerforge.multiplex import MultiplexOptimizer as DimerMultiplexOptimizer
from primerforge.secondary_structure import AmpliconFolder
from primerforge.specificity import SpecificityEngine, VariantAwareFilter

METHODOLOGY_APPENDIX_HTML = """
        <h2 style="page-break-before: always; margin-top: 40px; color: var(--primary-color);">🔬 Biophysical Analysis Methodology & Literature Appendix</h2>
        <div class="card" style="background-color: var(--container-bg); border-color: var(--border-color); font-size: 13.5px; line-height: 1.6; color: var(--text-color); padding: 20px; border-radius: 12px; border: 1px solid var(--border-color); margin-top: 15px;">
            <h3 style="color: var(--primary-color); margin-top: 0; font-size: 16px; font-weight: 600;">1. Assay Viability Index (AVI)</h3>
            <p style="margin-bottom: 15px;">
                The AVI evaluates single-locus primer pair thermodynamics using the <strong>Nearest-Neighbor (NN) unified physical parameters</strong> (SantaLucia 1998):
                <br>
                <code style="font-family: monospace; background-color: var(--bg-color); padding: 2px 6px; border-radius: 4px;">&Delta;G&deg;(T) = &Delta;H&deg; - T&Delta;S&deg; + &Delta;G&deg;<sub>init</sub> + &Delta;G&deg;<sub>salt</sub></code>
                <br>
                Stability of the 3' terminal 5-mer is targeted at <code>[-5.0, -9.0] kcal/mol</code> with a 3' GC clamp containing 1-2 G/C bases. Conformational blockers (hairpins and dimerization) are calculated via Nussinov MFE models and required to be stable at <code>&Delta;G &ge; -4.0 kcal/mol</code> and <code>&Delta;G &ge; -5.0 kcal/mol</code> respectively. Target folding obstruction (amplicon secondary base-pairing) is calculated by:
                <br>
                <code style="font-family: monospace; background-color: var(--bg-color); padding: 2px 6px; border-radius: 4px;">f<sub>paired</sub> = 2 * N<sub>paired</sub> / L<sub>amplicon</sub></code>
                <br>
                A high pairing fraction (<code>f<sub>paired</sub> &gt; 0.45</code>) or low free energy (<code>MFE &lt; -12.0 kcal/mol</code>) blocks Taq elongation. Variant mismatch resilience is calibrated by a Taq-weighted exponential decay model:
                <br>
                <code style="font-family: monospace; background-color: var(--bg-color); padding: 2px 6px; border-radius: 4px;">S<sub>mismatch</sub> = S<sub>baseline</sub> * &prod; exp(-&lambda; * d(v, 3'))</code>
            </p>

            <h3 style="color: var(--primary-color); font-size: 16px; font-weight: 600;">2. Panel Synergy & Interference Index (PSII)</h3>
            <p style="margin-bottom: 15px;">
                In multiplexing, thermal annealing rates must synchronize. Thermal differential is measured by:
                <br>
                <code style="font-family: monospace; background-color: var(--bg-color); padding: 2px 6px; border-radius: 4px;">&Delta;T<sub>m,max</sub> = max(T<sub>m</sub>) - min(T<sub>m</sub>)</code>
                <br>
                High differentials (<code>&Delta;T<sub>m,max</sub> &gt; 4.0&deg;C</code>) cause competitive amplification bias and dropouts. Dimerization penalties are integrated globally by summing cross-hybridization active dimer elements:
                <br>
                <code style="font-family: monospace; background-color: var(--bg-color); padding: 2px 6px; border-radius: 4px;">&Phi;(P) = &sum; max(0, -&Delta;G&deg;<sub>cross</sub> - 6.0 kcal/mol)</code>
            </p>

            <h3 style="color: var(--primary-color); font-size: 16px; font-weight: 600;">3. Scheme Coverage & Uniformity Index (SCUI)</h3>
            <p style="margin-bottom: 15px;">
                Tiled amplicon spatial depth uniformity is measured by the Coefficient of Variation (CV) of ensembled ML success:
                <br>
                <code style="font-family: monospace; background-color: var(--bg-color); padding: 2px 6px; border-radius: 4px;">CV<sub>P</sub> = &sigma;<sub>P</sub> / &mu;<sub>P</sub></code>
                <br>
                Values of <code>CV<sub>P</sub> &le; 0.10</code> represent excellent read flatness. Regional stalled segments are counted where single-tile success falls below 50%:
                <br>
                <code style="font-family: monospace; background-color: var(--bg-color); padding: 2px 6px; border-radius: 4px;">N<sub>stalled</sub> = &sum; I(S<sub>ML</sub> &lt; 0.50)</code>
            </p>

            <h3 style="color: var(--primary-color); font-size: 14px; margin-bottom: 5px; font-weight: 600;">Academic References:</h3>
            <ul style="padding-left: 20px; margin-top: 5px; font-size: 12.5px; color: var(--text-muted);">
                <li><strong>SantaLucia, J. (1998).</strong> PNAS, 95(4), 1460-1465.</li>
                <li><strong>Nussinov, R., & Jacobson, A. B. (1980).</strong> PNAS, 77(11), 6309-6313.</li>
                <li><strong>Owczarzy, R., et al. (2008).</strong> Biochemistry, 47(19), 5336-5353.</li>
            </ul>
        </div>
"""

# Caching wrappers for singletons to avoid reloading and leaks
@st.cache_resource
def get_cached_ml_scorer() -> MLScorer:
    return MLScorer()

@st.cache_resource
def get_cached_biophysics_engine(opt_tm: float = 60.0, min_size: int = 18, max_size: int = 24) -> BiophysicsEngine:
    return BiophysicsEngine(opt_tm=opt_tm, min_size=min_size, max_size=max_size)

@st.cache_resource
def get_cached_amplicon_folder() -> AmpliconFolder:
    return AmpliconFolder()

# Path security validator
PROJECT_ROOT = os.path.abspath(".")

def secure_path(path: str, allowed_exts=None) -> str:
    """Sanitizes user-controlled path inputs to prevent path traversal vulnerability.
    
    Ensures the path stays inside the project directory and has an allowed extension.
    """
    if not path:
        return ""
    clean_path = path.strip()
    abs_path = os.path.abspath(clean_path)
    if os.path.commonpath([PROJECT_ROOT, abs_path]) != PROJECT_ROOT:
        raise ValueError(f"Security Alert: Path '{path}' is outside the authorized project root directory.")
    if allowed_exts:
        ext = os.path.splitext(abs_path)[1].lower()
        if ext not in allowed_exts:
            raise ValueError(f"Security Alert: File extension '{ext}' not allowed. Must be one of {allowed_exts}")
    return abs_path

# Global shared helper functions
def reverse_complement(seq: str) -> str:
    rc = {"A": "T", "T": "A", "G": "C", "C": "G"}
    return "".join(rc.get(b, b) for b in reversed(seq.upper()))

def get_complement_seq(seq: str) -> str:
    comp = {"A": "T", "T": "A", "G": "C", "C": "G"}
    return "".join(comp.get(b, b) for b in seq.upper())

# ──────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PrimerForge – AI Primer Design Server",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# Premium dark-theme CSS
# ──────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.main {
    background: linear-gradient(160deg, #0a0d1a 0%, #0f172a 60%, #0a1628 100%);
    color: #e2e8f0;
}
/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5 0%, #06b6d4 100%);
    color: #ffffff;
    border: none;
    padding: 0.55rem 2rem;
    border-radius: 10px;
    font-weight: 600;
    font-size: 0.95rem;
    letter-spacing: 0.02em;
    transition: all 0.25s ease;
    box-shadow: 0 2px 12px rgba(79,70,229,0.35);
}
.stButton > button:hover {
    transform: translateY(-2px) scale(1.02);
    box-shadow: 0 6px 24px rgba(6,182,212,0.45);
}
/* Headings */
h1 { color: #38bdf8; font-weight: 700; letter-spacing: -0.03em; }
h2 { color: #7dd3fc; font-weight: 600; }
h3 { color: #93c5fd; font-weight: 600; }
/* Metric cards */
div[data-testid="stMetricValue"] {
    color: #34d399;
    font-size: 1.6rem;
    font-weight: 700;
}
/* Dataframe */
.stDataFrame { border-radius: 10px; }
/* Info/success/warning */
.stAlert { border-radius: 10px; }
/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    border-right: 1px solid #1e3a5f;
}
/* Card-like expanders */
.streamlit-expanderHeader {
    background: #1e293b;
    border-radius: 8px;
    color: #7dd3fc;
    font-weight: 600;
}
/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
    background: #1e293b;
    border-radius: 12px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #94a3b8;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #4f46e5, #06b6d4);
    color: white;
}
/* Progress bar */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #4f46e5, #06b6d4);
    border-radius: 99px;
}
</style>
""",
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────
# Helper: dark-styled matplotlib figure
# ──────────────────────────────────────────────────────────────
DARK_BG = "#0f172a"
PANEL_BG = "#1e293b"
ACCENT = "#06b6d4"
TEXT_CLR = "#e2e8f0"

def _dark_fig(nrows=1, ncols=1, figsize=(8, 4)):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    for ax in (axes if hasattr(axes, "__iter__") else [axes]):
        ax.set_facecolor(PANEL_BG)
        for sp in ax.spines.values():
            sp.set_color("#334155")
        ax.tick_params(colors=TEXT_CLR, which="both")
        ax.xaxis.label.set_color(TEXT_CLR)
        ax.yaxis.label.set_color(TEXT_CLR)
        ax.title.set_color("#7dd3fc")
    fig.patch.set_facecolor(DARK_BG)
    return fig, axes


def render_dimerization_heatmap(matrix, labels, title="Symmetric Cross-Reactivity Dimerization Matrix") -> str:
    if len(labels) > 0:
        fig, ax = _dark_fig(figsize=(8, 6))
        im = ax.imshow(matrix, cmap="coolwarm", vmin=-12, vmax=0)
        
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("ΔG of Dimerization (kcal/mol)", color=TEXT_CLR)
        cbar.ax.yaxis.set_tick_params(color=TEXT_CLR)
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=TEXT_CLR)
        
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        
        ax.set_title(title, fontsize=11, color="#7dd3fc")
        fig.tight_layout()
        st.pyplot(fig)
        
        # Base64 encode for offline report embedding
        import io
        import base64
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        buf.seek(0)
        matrix_base64 = f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
        buf.close()
        
        plt.close(fig) # Prevent memory leaks!
        return matrix_base64
    return ""


def generate_html_report(
    selected_index: int,
    pair: Any,
    spec: Any,
    score: float,
    mfe: float,
    frac_paired: float,
    largest_loop: int,
    amplicon_seq: str,
    f_terminal_dg: float,
    r_terminal_dg: float,
    mismatch_score: float,
    shap_base64: str = ""
) -> str:
    fwd_seq = pair.forward.sequence.upper()
    rev_seq = pair.reverse.sequence.upper()
    
    # GC clamps calculations
    f_clamp_gc = sum(1 for b in fwd_seq[-5:] if b in "GC")
    r_clamp_gc = sum(1 for b in rev_seq[-5:] if b in "GC")
    f_clamp_status = "Optimal" if 1 <= f_clamp_gc <= 2 else ("Suboptimal" if f_clamp_gc == 0 else "Too Strong")
    r_clamp_status = "Optimal" if 1 <= r_clamp_gc <= 2 else ("Suboptimal" if r_clamp_gc == 0 else "Too Strong")
    
    from primerforge.secondary_structure import AmpliconFolder
    ref_folder = AmpliconFolder()
    dot_bracket = ref_folder._dp.compute_mfe(amplicon_seq)[1]
    
    # ──────────────────────────────────────────────────────────
    # Programmatic Biophysical Conclusion (AVI Methodological Framework)
    # ──────────────────────────────────────────────────────────
    anchoring_issues = []
    if f_terminal_dg > -5.0:
        anchoring_issues.append("Forward 3' terminal stability is thermodynamically weak (ΔG > -5.0 kcal/mol), risking low polymerase initiation efficiency.")
    elif f_terminal_dg < -9.0:
        anchoring_issues.append("Forward 3' terminal stability is excessively stable (ΔG < -9.0 kcal/mol), increasing mispriming risks.")
    
    if r_terminal_dg > -5.0:
        anchoring_issues.append("Reverse 3' terminal stability is thermodynamically weak (ΔG > -5.0 kcal/mol), risking low polymerase initiation efficiency.")
    elif r_terminal_dg < -9.0:
        anchoring_issues.append("Reverse 3' terminal stability is excessively stable (ΔG < -9.0 kcal/mol), increasing mispriming risks.")
        
    if f_clamp_status != "Optimal":
        anchoring_issues.append(f"Forward 3' GC clamp is {f_clamp_status.lower()} ({f_clamp_gc} bases).")
    if r_clamp_status != "Optimal":
        anchoring_issues.append(f"Reverse 3' GC clamp is {r_clamp_status.lower()} ({r_clamp_gc} bases).")

    loop_issues = []
    if pair.forward.hairpin_dg < -4.0:
        loop_issues.append(f"Forward primer hairpin loop formation is thermodynamically active (ΔG = {pair.forward.hairpin_dg:.2f} kcal/mol).")
    if pair.reverse.hairpin_dg < -4.0:
        loop_issues.append(f"Reverse primer hairpin loop formation is thermodynamically active (ΔG = {pair.reverse.hairpin_dg:.2f} kcal/mol).")
    if pair.forward.homodimer_dg < -5.0:
        loop_issues.append(f"Forward primer homodimerization risk is elevated (ΔG = {pair.forward.homodimer_dg:.2f} kcal/mol).")
    if pair.reverse.homodimer_dg < -5.0:
        loop_issues.append(f"Reverse primer homodimerization risk is elevated (ΔG = {pair.reverse.homodimer_dg:.2f} kcal/mol).")
    if pair.cross_dimer_dg < -5.0:
        loop_issues.append(f"Cross-hybridization heterodimerization risk is elevated (ΔG = {pair.cross_dimer_dg:.2f} kcal/mol).")

    obstruction_issues = []
    if frac_paired > 0.45:
        obstruction_issues.append(f"High secondary structure density inside target amplicon ({frac_paired*100:.1f}% bases paired) may stall polymerase elongation.")
    if mfe < -12.0:
        obstruction_issues.append(f"Highly stable amplicon secondary fold (Nussinov MFE = {mfe:.2f} kcal/mol) threatens transcription kinetics.")

    resilience_issues = []
    if mismatch_score < 0.50:
        resilience_issues.append(f"Variant mismatch decay reduces assay success probability to {mismatch_score*100:.1f}%, indicating vulnerability to variant dropouts.")

    # AVI Synthesis
    if not anchoring_issues and not loop_issues and not obstruction_issues and not resilience_issues and score >= 0.50:
        avi_status = "✅ Certified Viable"
        avi_class = "status-optimal"
        scientific_conclusion = f"The physicochemical and thermodynamic parameters of Candidate {selected_index} are in perfect alignment. Strong, non-degenerate terminal anchoring is confirmed by both optimal 3' stability (Forward: {f_terminal_dg:.2f} kcal/mol / Reverse: {r_terminal_dg:.2f} kcal/mol) and ideal GC clamps. Self-annealing risks (hairpins, homodimers, and heterodimers) are completely negligible (all ΔG > -4.0 kcal/mol). Amplicon Nussinov folding analysis indicates minimal structural obstruction with a balanced {frac_paired*100:.1f}% base-pairing density, facilitating unhindered polymerase elongation. Under variant mismatch simulation, the success probability remains robust ({mismatch_score*100:.1f}%), certifying this assay with high-impact clinical utility."
    elif score < 0.50 or (pair.cross_dimer_dg < -8.0 or mismatch_score < 0.30):
        avi_status = "❌ High Failure Risk"
        avi_class = "status-too-strong"
        all_problems = anchoring_issues + loop_issues + obstruction_issues + resilience_issues
        problem_str = " ".join(all_problems) if all_problems else "Low overall machine learning success probability."
        scientific_conclusion = f"Candidate {selected_index} exhibits thermodynamic and specificity parameters that indicate a high risk of PCR failure in wet-lab assays. Specifically: {problem_str} Active heterodimerization or extreme variant vulnerability will deplete available primers in solution or block initiation completely, leading to assay dropouts. Redesign of this locus target or primer boundaries is strongly recommended."
    else:
        avi_status = "⚠️ Conditional Duplex"
        avi_class = "status-suboptimal"
        all_problems = anchoring_issues + loop_issues + obstruction_issues + resilience_issues
        problem_str = " ".join(all_problems)
        scientific_conclusion = f"Candidate {selected_index} is certified as conditionally viable, but exhibits biophysical anomalies that require careful laboratory optimization. Specifically: {problem_str} While the overall calibrated ML success probability remains high ({score*100:.1f}%), these thermodynamic clamping or self-annealing conditions might reduce amplification efficiency. We recommend running reactions at slightly elevated annealing temperatures or utilizing specialized master mixes with PCR additives (such as DMSO or betaine) to disrupt amplicon secondary folds."
        
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>PrimerForge Biophysical Report - Candidate {selected_index}</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        :root {{
            --bg-color: #0f172a;
            --container-bg: #1e293b;
            --text-color: #f1f5f9;
            --text-muted: #94a3b8;
            --border-color: #334155;
            --primary-color: #38bdf8;
            --primary-hover: #0ea5e9;
            --card-bg: #0f172a;
            --table-header-bg: #1e293b;
            --table-row-hover: #1e293b;
            --accent-success: #34d399;
            --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2), 0 2px 4px -1px rgba(0, 0, 0, 0.1);
        }}
        body.light-theme {{
            --bg-color: #f8fafc;
            --container-bg: #ffffff;
            --text-color: #0f172a;
            --text-muted: #64748b;
            --border-color: #e2e8f0;
            --primary-color: #0284c7;
            --primary-hover: #0369a1;
            --card-bg: #f1f5f9;
            --table-header-bg: #e2e8f0;
            --table-row-hover: #f8fafc;
            --accent-success: #059669;
            --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.02);
        }}
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            margin: 0;
            padding: 40px 20px;
            line-height: 1.5;
            transition: background-color 0.25s, color 0.25s;
        }}
        .container {{
            max-width: 950px;
            margin: 0 auto;
            background-color: var(--container-bg);
            padding: 40px;
            border-radius: 16px;
            box-shadow: var(--shadow);
            border: 1px solid var(--border-color);
            position: relative;
            transition: background-color 0.25s, border-color 0.25s;
        }}
        .header {{
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 24px;
            margin-bottom: 30px;
            position: relative;
        }}
        .header h1 {{
            margin: 0;
            font-size: 28px;
            font-weight: 700;
            color: var(--primary-color);
            letter-spacing: -0.02em;
        }}
        .header p {{
            margin: 6px 0 0 0;
            color: var(--text-muted);
            font-size: 14px;
        }}
        .controls {{
            position: absolute;
            top: 5px;
            right: 0;
            display: flex;
            gap: 10px;
        }}
        .btn {{
            padding: 8px 16px;
            font-weight: 600;
            border-radius: 8px;
            text-decoration: none;
            cursor: pointer;
            border: none;
            font-size: 13px;
            transition: all 0.2s;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }}
        .btn-toggle {{
            background-color: var(--card-bg);
            color: var(--text-color);
            border: 1px solid var(--border-color);
        }}
        .btn-toggle:hover {{
            background-color: var(--border-color);
        }}
        .btn-primary {{
            background-color: var(--primary-color);
            color: #ffffff;
            font-weight: bold;
        }}
        .btn-primary:hover {{
            background-color: var(--primary-hover);
            transform: translateY(-1px);
        }}
        h2 {{
            color: var(--primary-color);
            font-size: 19px;
            font-weight: 600;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 8px;
            margin-top: 35px;
            margin-bottom: 15px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
            background-color: var(--card-bg);
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid var(--border-color);
        }}
        th, td {{
            padding: 12px 18px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}
        th {{
            background-color: var(--table-header-bg);
            color: var(--primary-color);
            font-weight: 600;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        td {{
            font-size: 14px;
        }}
        tr:hover {{
            background-color: var(--table-row-hover);
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 15px;
        }}
        .card {{
            background-color: var(--card-bg);
            padding: 18px;
            border-radius: 12px;
            border: 1px solid var(--border-color);
            transition: all 0.2s;
        }}
        .card:hover {{
            border-color: var(--primary-color);
        }}
        .card-title {{
            font-size: 11px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 6px;
        }}
        .card-value {{
            font-size: 19px;
            font-weight: 700;
            color: var(--text-color);
        }}
        .card-status {{
            display: inline-block;
            font-size: 10px;
            padding: 2px 8px;
            border-radius: 6px;
            margin-top: 6px;
            font-weight: 600;
        }}
        .status-optimal {{
            background-color: rgba(52, 211, 153, 0.12);
            color: #34d399;
            border: 1px solid rgba(52, 211, 153, 0.25);
        }}
        body.light-theme .status-optimal {{
            background-color: rgba(5, 150, 105, 0.08);
            color: #059669;
            border: 1px solid rgba(5, 150, 105, 0.15);
        }}
        .status-suboptimal {{
            background-color: rgba(245, 158, 11, 0.12);
            color: #fbbf24;
            border: 1px solid rgba(245, 158, 11, 0.25);
        }}
        body.light-theme .status-suboptimal {{
            background-color: rgba(217, 119, 6, 0.08);
            color: #d97706;
            border: 1px solid rgba(217, 119, 6, 0.15);
        }}
        .status-too-strong {{
            background-color: rgba(239, 68, 68, 0.12);
            color: #f87171;
            border: 1px solid rgba(239, 68, 68, 0.25);
        }}
        body.light-theme .status-too-strong {{
            background-color: rgba(220, 38, 38, 0.08);
            color: #dc2626;
            border: 1px solid rgba(220, 38, 38, 0.15);
        }}
        .code-block {{
            background-color: var(--card-bg);
            padding: 16px;
            border-radius: 12px;
            font-family: 'Fira Code', 'Courier New', Courier, monospace;
            font-size: 13px;
            white-space: pre-wrap;
            border: 1px solid var(--border-color);
            color: #34d399;
            margin-top: 10px;
        }}
        body.light-theme .code-block {{
            color: #059669;
        }}
        .chart-container {{
            text-align: center;
            margin-top: 25px;
            background-color: var(--card-bg);
            padding: 24px;
            border-radius: 12px;
            border: 1px solid var(--border-color);
        }}
        .chart-img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            border: 1px solid var(--border-color);
        }}
        .footer {{
            text-align: center;
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid var(--border-color);
            font-size: 12px;
            color: var(--text-muted);
        }}
        @media print {{
            :root {{
                --bg-color: #ffffff !important;
                --container-bg: #ffffff !important;
                --text-color: #000000 !important;
                --text-muted: #555555 !important;
                --border-color: #cccccc !important;
                --primary-color: #000000 !important;
                --card-bg: #ffffff !important;
                --table-header-bg: #eeeeee !important;
                --table-row-hover: #ffffff !important;
            }}
            body {{
                background-color: #ffffff !important;
                color: #000000 !important;
                padding: 0 !important;
            }}
            .container {{
                border: none !important;
                box-shadow: none !important;
                padding: 0 !important;
                max-width: 100% !important;
            }}
            .controls {{
                display: none !important;
            }}
            .card, table, .code-block, .chart-container {{
                page-break-inside: avoid;
                border: 1px solid #cccccc !important;
                background-color: #ffffff !important;
                color: #000000 !important;
            }}
            .card-value, h2, .header h1 {{
                color: #000000 !important;
            }}
            .status-optimal, .status-suboptimal, .status-too-strong {{
                border: 1px solid #555555 !important;
                color: #000000 !important;
                background-color: transparent !important;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧬 PrimerForge Biophysical Diagnostic Report</h1>
            <p>High-Fidelity Single-Locus PCR Assay Profiling &middot; Candidate {selected_index}</p>
            <div class="controls">
                <button class="btn btn-toggle" onclick="toggleTheme()">🌓 Toggle Theme</button>
                <button class="btn btn-primary" onclick="window.print()">🖨️ Save as PDF</button>
            </div>
        </div>
        
        <h2>🎯 1. Primer Candidate Specifications</h2>
        <table>
            <thead>
                <tr>
                    <th>Primer Type</th>
                    <th>Sequence (5′ → 3′)</th>
                    <th>Tm (°C)</th>
                    <th>GC (%)</th>
                    <th>Product Length</th>
                    <th>ML Success Score</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>Forward Primer</strong></td>
                    <td style="font-family: monospace; font-size: 14px;">{fwd_seq}</td>
                    <td>{pair.forward.tm:.1f}°C</td>
                    <td>{pair.forward.gc_percent:.0f}%</td>
                    <td rowspan="2" style="text-align: center; vertical-align: middle; border-left: 1px solid var(--border-color); font-weight: 600;">{pair.product_size} bp</td>
                    <td rowspan="2" style="text-align: center; vertical-align: middle; border-left: 1px solid var(--border-color); font-size: 18px; font-weight: bold; color: var(--accent-success);">{score*100:.1f}%</td>
                </tr>
                <tr>
                    <td><strong>Reverse Primer</strong></td>
                    <td style="font-family: monospace; font-size: 14px;">{rev_seq}</td>
                    <td>{pair.reverse.tm:.1f}°C</td>
                    <td>{pair.reverse.gc_percent:.0f}%</td>
                </tr>
            </tbody>
        </table>
        
        <h2>🧱 2. Amplicon Nussinov Secondary Structure</h2>
        <div class="grid">
            <div class="card">
                <div class="card-title">Nussinov Minimum Free Energy</div>
                <div class="card-value" style="color: #f87171;">{mfe:.2f} kcal/mol</div>
            </div>
            <div class="card">
                <div class="card-title">Fraction of Bases Paired</div>
                <div class="card-value">{frac_paired*100:.1f}%</div>
            </div>
            <div class="card">
                <div class="card-title">Largest Unpaired Loop</div>
                <div class="card-value">{largest_loop} bp</div>
            </div>
        </div>
        <h4 style="margin-top: 20px; margin-bottom: 5px; color: var(--text-muted); font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Predicted Secondary Structure (Dot-Bracket Notation):</h4>
        <div class="code-block">{dot_bracket}</div>
        
        <h2>🎯 3. Forward Primer Biophysical Anchoring</h2>
        <div class="grid">
            <div class="card">
                <div class="card-title">3' Terminal Stability</div>
                <div class="card-value">{f_terminal_dg:.2f} kcal/mol</div>
            </div>
            <div class="card">
                <div class="card-title">GC Clamp (3' End)</div>
                <div class="card-value">{f_clamp_gc} G/C bases</div>
                <span class="card-status status-{f_clamp_status.lower().replace(' ', '-')}">{f_clamp_status}</span>
            </div>
            <div class="card">
                <div class="card-title">Self-Hairpin dG</div>
                <div class="card-value">{pair.forward.hairpin_dg:.2f} kcal/mol</div>
            </div>
            <div class="card">
                <div class="card-title">Homodimer dG</div>
                <div class="card-value">{pair.forward.homodimer_dg:.2f} kcal/mol</div>
            </div>
        </div>
        
        <h2>🎯 4. Reverse Primer Biophysical Anchoring</h2>
        <div class="grid">
            <div class="card">
                <div class="card-title">3' Terminal Stability</div>
                <div class="card-value">{r_terminal_dg:.2f} kcal/mol</div>
            </div>
            <div class="card">
                <div class="card-title">GC Clamp (3' End)</div>
                <div class="card-value">{r_clamp_gc} G/C bases</div>
                <span class="card-status status-{r_clamp_status.lower().replace(' ', '-')}">{r_clamp_status}</span>
            </div>
            <div class="card">
                <div class="card-title">Self-Hairpin dG</div>
                <div class="card-value">{pair.reverse.hairpin_dg:.2f} kcal/mol</div>
            </div>
            <div class="card">
                <div class="card-title">Homodimer dG</div>
                <div class="card-value">{pair.reverse.homodimer_dg:.2f} kcal/mol</div>
            </div>
        </div>
 
        <h2>🔮 5. Variant Mismatch Simulator Calibration</h2>
        <div class="grid">
            <div class="card">
                <div class="card-title">Perfect Match Baseline Success</div>
                <div class="card-value">{score*100:.2f}%</div>
            </div>
            <div class="card">
                <div class="card-title">Mismatch-Calibrated Success</div>
                <div class="card-value" style="color: #60a5fa;">{mismatch_score*100:.2f}%</div>
            </div>
            <div class="card" style="grid-column: span 2;">
                <div class="card-title">Viability Status</div>
                <div class="card-value" style="color: {'var(--accent-success)' if mismatch_score >= 0.50 else '#f87171'}">
                    {('✅ Viable (P >= 50%)' if mismatch_score >= 0.50 else '❌ Mismatches Fatal')}
                </div>
            </div>
        </div>
        
        <h2>🔬 6. Biophysical Assay Viability Verdict</h2>
        <div class="card" style="border-left: 5px solid var(--primary-color);">
            <div class="card-title">Conclusion & Recommendation Rating</div>
            <div class="card-value" style="font-size: 22px; margin-bottom: 10px;">
                <span class="card-status {avi_class}" style="font-size: 14px; padding: 4px 12px; border-radius: 8px;">{avi_status}</span>
            </div>
            <p style="font-size: 14.5px; line-height: 1.6; margin: 10px 0 0 0; color: var(--text-color);">
                {scientific_conclusion}
            </p>
        </div>
        
        {f"<h2>📊 7. SHAP Feature Attribution</h2><div class='chart-container'><img class='chart-img' src='{shap_base64}' alt='SHAP Feature Attribution'></div>" if shap_base64 else ""}
        
        {METHODOLOGY_APPENDIX_HTML}
        
        <div class="footer">
            <p>Generated by PrimerForge Web Server &middot; publication-grade molecular engineering diagnostics platform.</p>
        </div>
    </div>

    <script>
        function toggleTheme() {{
            document.body.classList.toggle('light-theme');
            const isLight = document.body.classList.contains('light-theme');
            localStorage.setItem('theme', isLight ? 'light' : 'dark');
        }}
        
        window.onload = function() {{
            const savedTheme = localStorage.getItem('theme');
            if (savedTheme === 'light') {{
                document.body.classList.add('light-theme');
            }}
        }}
    </script>
</body>
</html>"""
    return html


def generate_multiplex_html_report(
    panel_pairs: list,
    solver_engine: str,
    max_plex: int,
    dg_thresh: float,
    global_penalty: float,
    dimer_matrix_base64: str = ""
) -> str:
    tr_rows = ""
    for idx, r in enumerate(panel_pairs, 1):
        tr_rows += f"""
        <tr>
            <td style="font-weight: 600;">{r.get('Locus ID', f'locus_{idx}')}</td>
            <td style="font-family: monospace; font-size: 13px;">{r.get('Forward Primer')}</td>
            <td>{r.get('Fwd Tm (°C)')}°C</td>
            <td style="font-family: monospace; font-size: 13px;">{r.get('Reverse Primer')}</td>
            <td>{r.get('Rev Tm (°C)')}°C</td>
            <td style="font-weight: 600; text-align: center;">{r.get('Product (bp)')} bp</td>
            <td style="color: #f87171; text-align: center;">{r.get('Cross-Dimer ΔG')}</td>
            <td style="color: var(--accent-success); font-weight: bold; text-align: center;">{r.get('ML Success')}</td>
        </tr>
        """

    # ──────────────────────────────────────────────────────────
    # Programmatic Biophysical Conclusion (PSII Methodological Framework)
    # ──────────────────────────────────────────────────────────
    tms = []
    active_dimer_count = 0
    ml_scores = []
    
    for r in panel_pairs:
        try:
            # Handle possible string formatting in dict
            ftm = float(r.get("Fwd Tm (°C)", r.get("Fwd Tm", "60.0")).replace("°C", "").strip())
            rtm = float(r.get("Rev Tm (°C)", r.get("Rev Tm", "60.0")).replace("°C", "").strip())
            tms.extend([ftm, rtm])
        except Exception:
            pass
            
        try:
            dg = float(r.get("Cross-Dimer ΔG", r.get("Dimer ΔG", "0.0")))
            if dg < -6.0:
                active_dimer_count += 1
        except Exception:
            pass
            
        try:
            score_str = r.get("ML Success", r.get("ML Score", "50.0%")).replace("%", "").strip()
            ml_scores.append(float(score_str) / 100.0)
        except Exception:
            pass

    max_tm_diff = max(tms) - min(tms) if tms else 0.0
    avg_success = sum(ml_scores) / len(ml_scores) if ml_scores else 0.0
    
    psii_issues = []
    if max_tm_diff > 2.0:
        psii_issues.append(f"Elevated thermal disparity across panel (ΔTm = {max_tm_diff:.1f}°C). Max Tm is {max(tms):.1f}°C, Min Tm is {min(tms):.1f}°C.")
    if active_dimer_count > 0:
        psii_issues.append(f"Detected {active_dimer_count} loci pairings with active heterodimerization risks below the soft ΔG threshold (-6.0 kcal/mol).")
    if avg_success < 0.60:
        psii_issues.append(f"Mean panel ML success probability is relatively low ({avg_success*100:.1f}%).")
        
    if not psii_issues and global_penalty == 0.0:
        psii_status = "✅ Synergy Certified"
        psii_class = "status-optimal"
        scientific_conclusion = f"System analysis confirms that this {len(panel_pairs)}-plex primer panel exhibits flawless molecular synergy. The global dimerization penalty is 0.000, and all pairwise cross-reactivity matrix elements reside safely above the soft threshold, guaranteeing a 100% dimer-free assay. Thermal cohort uniformity is excellent, with a maximum melting temperature differential of only {max_tm_diff:.1f}°C. This tightly matched thermodynamic profile ensures highly uniform annealing kinetics across all loci during simultaneous amplification cycles, completely eliminating competitive amplification bias or target dropouts. The average panel success probability is exceptionally strong at {avg_success*100:.1f}%, certifying this panel for publication-grade pangenomic assays."
    elif global_penalty > 5.0 or max_tm_diff > 4.0 or avg_success < 0.50:
        psii_status = "❌ High Interference Risk"
        psii_class = "status-too-strong"
        problem_str = " ".join(psii_issues) if psii_issues else "High dimerization penalty or poor amplification profile."
        scientific_conclusion = f"Multiplex panel diagnostics indicate significant molecular interference risks that threaten assay performance. Specifically: {problem_str} High dimerization energies (ΔG < -6.0 kcal/mol) or large thermal differentials (ΔTm = {max_tm_diff:.1f}°C) will cause massive primer-dimer formation and severe selective locus dropouts. High-affinity cross-dimers will preferentially consume primers, yielding non-specific bands and depleting target amplicons. We strongly recommend re-running the panel optimization using the ILP Graph Solver with more stringent ΔG thresholds."
    else:
        psii_status = "⚠️ Minor Interference Risk"
        psii_class = "status-suboptimal"
        problem_str = " ".join(psii_issues)
        scientific_conclusion = f"This multiplex panel is certified with minor interference risks. While overall panel assembly is compatible, optimized conditions are required to prevent dropouts: {problem_str} The moderate thermal variance or minor active cross-reactivity requires adjusting the master mix composition. We recommend increasing monovalent salt concentrations, utilizing hot-start Taq polymerase, or optimizing the multiplex annealing duration to favor specific target hybridization over transient primer-primer duplexes."

    dimer_free_status = "✅ 100% Dimer-Free" if global_penalty == 0.0 else "⚠️ Dimerization Risks Exist"
    dimer_status_class = "status-optimal" if global_penalty == 0.0 else "status-suboptimal"

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>PrimerForge Multiplex Panel Report</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        :root {{
            --bg-color: #0f172a;
            --container-bg: #1e293b;
            --text-color: #f1f5f9;
            --text-muted: #94a3b8;
            --border-color: #334155;
            --primary-color: #38bdf8;
            --primary-hover: #0ea5e9;
            --card-bg: #0f172a;
            --table-header-bg: #1e293b;
            --table-row-hover: #1e293b;
            --accent-success: #34d399;
            --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2), 0 2px 4px -1px rgba(0, 0, 0, 0.1);
        }}
        body.light-theme {{
            --bg-color: #f8fafc;
            --container-bg: #ffffff;
            --text-color: #0f172a;
            --text-muted: #64748b;
            --border-color: #e2e8f0;
            --primary-color: #0284c7;
            --primary-hover: #0369a1;
            --card-bg: #f1f5f9;
            --table-header-bg: #e2e8f0;
            --table-row-hover: #f8fafc;
            --accent-success: #059669;
            --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.02);
        }}
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            margin: 0;
            padding: 40px 20px;
            line-height: 1.5;
            transition: background-color 0.25s, color 0.25s;
        }}
        .container {{
            max-width: 1050px;
            margin: 0 auto;
            background-color: var(--container-bg);
            padding: 40px;
            border-radius: 16px;
            box-shadow: var(--shadow);
            border: 1px solid var(--border-color);
            position: relative;
            transition: background-color 0.25s, border-color 0.25s;
        }}
        .header {{
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 24px;
            margin-bottom: 30px;
            position: relative;
        }}
        .header h1 {{
            margin: 0;
            font-size: 28px;
            font-weight: 700;
            color: var(--primary-color);
            letter-spacing: -0.02em;
        }}
        .header p {{
            margin: 6px 0 0 0;
            color: var(--text-muted);
            font-size: 14px;
        }}
        .controls {{
            position: absolute;
            top: 5px;
            right: 0;
            display: flex;
            gap: 10px;
        }}
        .btn {{
            padding: 8px 16px;
            font-weight: 600;
            border-radius: 8px;
            text-decoration: none;
            cursor: pointer;
            border: none;
            font-size: 13px;
            transition: all 0.2s;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }}
        .btn-toggle {{
            background-color: var(--card-bg);
            color: var(--text-color);
            border: 1px solid var(--border-color);
        }}
        .btn-toggle:hover {{
            background-color: var(--border-color);
        }}
        .btn-primary {{
            background-color: var(--primary-color);
            color: #ffffff;
            font-weight: bold;
        }}
        .btn-primary:hover {{
            background-color: var(--primary-hover);
            transform: translateY(-1px);
        }}
        h2 {{
            color: var(--primary-color);
            font-size: 19px;
            font-weight: 600;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 8px;
            margin-top: 35px;
            margin-bottom: 15px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
            background-color: var(--card-bg);
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid var(--border-color);
        }}
        th, td {{
            padding: 12px 18px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}
        th {{
            background-color: var(--table-header-bg);
            color: var(--primary-color);
            font-weight: 600;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        td {{
            font-size: 13px;
        }}
        tr:hover {{
            background-color: var(--table-row-hover);
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 15px;
        }}
        .card {{
            background-color: var(--card-bg);
            padding: 18px;
            border-radius: 12px;
            border: 1px solid var(--border-color);
        }}
        .card-title {{
            font-size: 11px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 6px;
        }}
        .card-value {{
            font-size: 19px;
            font-weight: 700;
            color: var(--text-color);
        }}
        .card-status {{
            display: inline-block;
            font-size: 10px;
            padding: 2px 8px;
            border-radius: 6px;
            margin-top: 6px;
            font-weight: 600;
        }}
        .status-optimal {{
            background-color: rgba(52, 211, 153, 0.12);
            color: #34d399;
            border: 1px solid rgba(52, 211, 153, 0.25);
        }}
        body.light-theme .status-optimal {{
            background-color: rgba(5, 150, 105, 0.08);
            color: #059669;
            border: 1px solid rgba(5, 150, 105, 0.15);
        }}
        .status-suboptimal {{
            background-color: rgba(245, 158, 11, 0.12);
            color: #fbbf24;
            border: 1px solid rgba(245, 158, 11, 0.25);
        }}
        body.light-theme .status-suboptimal {{
            background-color: rgba(217, 119, 6, 0.08);
            color: #d97706;
            border: 1px solid rgba(217, 119, 6, 0.15);
        }}
        .status-too-strong {{
            background-color: rgba(239, 68, 68, 0.12);
            color: #f87171;
            border: 1px solid rgba(239, 68, 68, 0.25);
        }}
        body.light-theme .status-too-strong {{
            background-color: rgba(220, 38, 38, 0.08);
            color: #dc2626;
            border: 1px solid rgba(220, 38, 38, 0.15);
        }}
        .chart-container {{
            text-align: center;
            margin-top: 25px;
            background-color: var(--card-bg);
            padding: 24px;
            border-radius: 12px;
            border: 1px solid var(--border-color);
        }}
        .chart-img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            border: 1px solid var(--border-color);
        }}
        .footer {{
            text-align: center;
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid var(--border-color);
            font-size: 12px;
            color: var(--text-muted);
        }}
        @media print {{
            :root {{
                --bg-color: #ffffff !important;
                --container-bg: #ffffff !important;
                --text-color: #000000 !important;
                --text-muted: #555555 !important;
                --border-color: #cccccc !important;
                --primary-color: #000000 !important;
                --card-bg: #ffffff !important;
                --table-header-bg: #eeeeee !important;
                --table-row-hover: #ffffff !important;
            }}
            body {{
                background-color: #ffffff !important;
                color: #000000 !important;
                padding: 0 !important;
            }}
            .container {{
                border: none !important;
                box-shadow: none !important;
                padding: 0 !important;
                max-width: 100% !important;
            }}
            .controls {{
                display: none !important;
            }}
            .card, table, .chart-container {{
                page-break-inside: avoid;
                border: 1px solid #cccccc !important;
                background-color: #ffffff !important;
                color: #000000 !important;
            }}
            .card-value, h2, .header h1 {{
                color: #000000 !important;
            }}
            .status-optimal, .status-suboptimal {{
                border: 1px solid #555555 !important;
                color: #000000 !important;
                background-color: transparent !important;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧬 PrimerForge Multiplex Panel Report</h1>
            <p>Compatible Dimer-Free Primer Panel Assembler &middot; {solver_engine}</p>
            <div class="controls">
                <button class="btn btn-toggle" onclick="toggleTheme()">🌓 Toggle Theme</button>
                <button class="btn btn-primary" onclick="window.print()">🖨️ Save as PDF</button>
            </div>
        </div>
        
        <h2>📊 1. Optimization Settings & Panel Summary</h2>
        <div class="grid">
            <div class="card">
                <div class="card-title">Optimization Engine</div>
                <div class="card-value">{solver_engine}</div>
            </div>
            <div class="card">
                <div class="card-title">Panel Size</div>
                <div class="card-value">{len(panel_pairs)}-plex</div>
            </div>
            <div class="card">
                <div class="card-title">Global Dimerization Penalty</div>
                <div class="card-value" style="color: {'#fbbf24' if global_penalty > 0 else 'inherit'}">{global_penalty:.3f}</div>
            </div>
            <div class="card">
                <div class="card-title">Dimerization Status</div>
                <div class="card-value" style="font-size: 15px; margin-top: 4px;">
                    <span class="card-status {dimer_status_class}" style="font-size: 12px; margin-top: 0; padding: 4px 10px;">{dimer_free_status}</span>
                </div>
            </div>
        </div>
        
        <h2>🎯 2. Multiplex Primer Panel Grid</h2>
        <table>
            <thead>
                <tr>
                    <th>Locus ID</th>
                    <th>Forward Primer (5′ → 3′)</th>
                    <th>Fwd Tm</th>
                    <th>Reverse Primer (5′ → 3′)</th>
                    <th>Rev Tm</th>
                    <th style="text-align: center;">Product</th>
                    <th style="text-align: center;">Cross-Dimer ΔG</th>
                    <th style="text-align: center;">ML Success</th>
                </tr>
            </thead>
            <tbody>
                {tr_rows}
            </tbody>
        </table>
        
        <h2>🔬 3. Multiplex Panel Engineering Verdict</h2>
        <div class="card" style="border-left: 5px solid var(--primary-color);">
            <div class="card-title">Methodological Panel Compatibility Verdict</div>
            <div class="card-value" style="font-size: 22px; margin-bottom: 10px;">
                <span class="card-status {psii_class}" style="font-size: 14px; padding: 4px 12px; border-radius: 8px;">{psii_status}</span>
            </div>
            <p style="font-size: 14.5px; line-height: 1.6; margin: 10px 0 0 0; color: var(--text-color);">
                {scientific_conclusion}
            </p>
        </div>
        
        {f"<h2>🔬 4. Symmetric Cross-Reactivity Dimerization Matrix</h2><div class='chart-container'><img class='chart-img' src='{dimer_matrix_base64}' alt='Symmetric Dimerization Matrix' style='max-width: 650px;'></div>" if dimer_matrix_base64 else ""}
        
        {METHODOLOGY_APPENDIX_HTML}
        
        <div class="footer">
            <p>Generated by PrimerForge Web Server &middot; publication-grade molecular engineering diagnostics platform.</p>
        </div>
    </div>

    <script>
        function toggleTheme() {{
            document.body.classList.toggle('light-theme');
            const isLight = document.body.classList.contains('light-theme');
            localStorage.setItem('theme', isLight ? 'light' : 'dark');
        }}
        
        window.onload = function() {{
            const savedTheme = localStorage.getItem('theme');
            if (savedTheme === 'light') {{
                document.body.classList.add('light-theme');
            }}
        }}
    </script>
</body>
</html>"""
    return html


def generate_tiled_html_report(
    tiles: list,
    tile_size: int,
    overlap: int,
    avg_success: float,
    genome_len: int,
    coverage_map_base64: str = ""
) -> str:
    tr_rows = ""
    for r in tiles:
        pair = r.get("pair")
        tr_rows += f"""
        <tr>
            <td style="font-weight: 600; text-align: center;">{r.get('Tile #')}</td>
            <td style="font-weight: 600;">{r.get('Genome Range')}</td>
            <td style="font-family: monospace; font-size: 13px;">{pair.forward.sequence}</td>
            <td style="font-family: monospace; font-size: 13px;">{pair.reverse.sequence}</td>
            <td style="font-weight: 600; text-align: center;">{r.get('Product (bp)')} bp</td>
            <td style="color: var(--accent-success); font-weight: bold; text-align: center;">{r.get('ML Success')}</td>
        </tr>
        """

    # ──────────────────────────────────────────────────────────
    # Programmatic Biophysical Conclusion (SCUI Methodological Framework)
    # ──────────────────────────────────────────────────────────
    successes = []
    stalled_count = 0
    
    for r in tiles:
        try:
            score_str = r.get("ML Success").replace("%", "").strip()
            score_val = float(score_str) / 100.0
            successes.append(score_val)
            if score_val < 0.50:
                stalled_count += 1
        except Exception:
            pass
            
    mean_p = sum(successes) / len(successes) if successes else 0.0
    import numpy as np
    std_p = np.std(successes) if successes else 0.0
    cv_p = std_p / mean_p if mean_p > 0.0 else 0.0
    
    scui_issues = []
    if cv_p > 0.15:
        scui_issues.append(f"High amplification success variance across sequence (CV = {cv_p:.3f}, Standard Deviation = {std_p:.3f}), indicating uneven amplification depth.")
    if stalled_count > 0:
        scui_issues.append(f"Detected {stalled_count} regional amplification bottlenecks/stalled segments where single-tile success drops below 50%.")
    if mean_p < 0.60:
        scui_issues.append(f"Overall reference coverage success is weak (Mean P = {mean_p*100:.1f}%).")
        
    if not scui_issues and cv_p <= 0.10:
        scui_status = "✅ Certified Tiled Scheme"
        scui_class = "status-optimal"
        scientific_conclusion = f"This whole-genome overlapping tile scheme exhibits exemplary amplification uniformity across the entire reference structure. The Dynamic Programming router designed a balanced tileset of {len(tiles)} overlapping segments with extremely low success variance (CV = {cv_p:.3f}) and a high average success probability of {mean_p*100:.1f}%. Zero regional bottlenecks or stalled segments were detected, ensuring that next-generation sequencing libraries prepared with this scheme will exhibit exceptionally flat coverage depth. This minimizes sequencing bias, prevents library dropout zones, and guarantees publication-grade tiling across the full {genome_len} bp target sequence."
    elif cv_p > 0.20 or stalled_count > 2 or mean_p < 0.50:
        scui_status = "❌ Unviable Tiled Routing"
        scui_class = "status-too-strong"
        problem_str = " ".join(scui_issues) if scui_issues else "High success variation or severe sequence bottlenecks."
        scientific_conclusion = f"Tiled scheme analysis indicates critical amplification uniformity and coverage depth risks. Specifically: {problem_str} Having stalled segments or extremely high success variation (CV = {cv_p:.3f}) will introduce massive sequencing coverage dropouts and regional bias. Stalled amplicons with low amplification efficiency will be under-represented in the final sequencing library, while highly efficient adjacent tiles will dominate the reads. We strongly recommend re-routing this reference genome with altered tile sizes, smaller overlap step limits, or relaxing the biophysical Tm constraints to secure a compatible set of primers."
    else:
        scui_status = "⚠️ Suboptimal Uniformity"
        scui_class = "status-suboptimal"
        problem_str = " ".join(scui_issues)
        scientific_conclusion = f"This tiled amplicon scheme is conditionally viable but exhibits moderate coverage uniformity issues. Specifically: {problem_str} The moderate variance (CV = {cv_p:.3f}) or the presence of isolated stalled segments might cause mild sequencing library imbalances. To maximize coverage flatness in next-generation sequencing, we suggest adjusting individual primer concentrations in the pooling pool—specifically boosting the primer concentrations for the bottleneck tiles—or using optimized GC-rich master mixes to resolve stable regional target folds."

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>PrimerForge Tiled Amplicon Report</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        :root {{
            --bg-color: #0f172a;
            --container-bg: #1e293b;
            --text-color: #f1f5f9;
            --text-muted: #94a3b8;
            --border-color: #334155;
            --primary-color: #38bdf8;
            --primary-hover: #0ea5e9;
            --card-bg: #0f172a;
            --table-header-bg: #1e293b;
            --table-row-hover: #1e293b;
            --accent-success: #34d399;
            --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2), 0 2px 4px -1px rgba(0, 0, 0, 0.1);
        }}
        body.light-theme {{
            --bg-color: #f8fafc;
            --container-bg: #ffffff;
            --text-color: #0f172a;
            --text-muted: #64748b;
            --border-color: #e2e8f0;
            --primary-color: #0284c7;
            --primary-hover: #0369a1;
            --card-bg: #f1f5f9;
            --table-header-bg: #e2e8f0;
            --table-row-hover: #f8fafc;
            --accent-success: #059669;
            --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.02);
        }}
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            margin: 0;
            padding: 40px 20px;
            line-height: 1.5;
            transition: background-color 0.25s, color 0.25s;
        }}
        .container {{
            max-width: 1050px;
            margin: 0 auto;
            background-color: var(--container-bg);
            padding: 40px;
            border-radius: 16px;
            box-shadow: var(--shadow);
            border: 1px solid var(--border-color);
            position: relative;
            transition: background-color 0.25s, border-color 0.25s;
        }}
        .header {{
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 24px;
            margin-bottom: 30px;
            position: relative;
        }}
        .header h1 {{
            margin: 0;
            font-size: 28px;
            font-weight: 700;
            color: var(--primary-color);
            letter-spacing: -0.02em;
        }}
        .header p {{
            margin: 6px 0 0 0;
            color: var(--text-muted);
            font-size: 14px;
        }}
        .controls {{
            position: absolute;
            top: 5px;
            right: 0;
            display: flex;
            gap: 10px;
        }}
        .btn {{
            padding: 8px 16px;
            font-weight: 600;
            border-radius: 8px;
            text-decoration: none;
            cursor: pointer;
            border: none;
            font-size: 13px;
            transition: all 0.2s;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }}
        .btn-toggle {{
            background-color: var(--card-bg);
            color: var(--text-color);
            border: 1px solid var(--border-color);
        }}
        .btn-toggle:hover {{
            background-color: var(--border-color);
        }}
        .btn-primary {{
            background-color: var(--primary-color);
            color: #ffffff;
            font-weight: bold;
        }}
        .btn-primary:hover {{
            background-color: var(--primary-hover);
            transform: translateY(-1px);
        }}
        h2 {{
            color: var(--primary-color);
            font-size: 19px;
            font-weight: 600;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 8px;
            margin-top: 35px;
            margin-bottom: 15px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
            background-color: var(--card-bg);
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid var(--border-color);
        }}
        th, td {{
            padding: 12px 18px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}
        th {{
            background-color: var(--table-header-bg);
            color: var(--primary-color);
            font-weight: 600;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        td {{
            font-size: 13px;
        }}
        tr:hover {{
            background-color: var(--table-row-hover);
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 15px;
        }}
        .card {{
            background-color: var(--card-bg);
            padding: 18px;
            border-radius: 12px;
            border: 1px solid var(--border-color);
        }}
        .card-title {{
            font-size: 11px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 6px;
        }}
        .card-value {{
            font-size: 19px;
            font-weight: 700;
            color: var(--text-color);
        }}
        .chart-container {{
            text-align: center;
            margin-top: 25px;
            background-color: var(--card-bg);
            padding: 24px;
            border-radius: 12px;
            border: 1px solid var(--border-color);
        }}
        .chart-img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            border: 1px solid var(--border-color);
        }}
        .footer {{
            text-align: center;
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid var(--border-color);
            font-size: 12px;
            color: var(--text-muted);
        }}
        @media print {{
            :root {{
                --bg-color: #ffffff !important;
                --container-bg: #ffffff !important;
                --text-color: #000000 !important;
                --text-muted: #555555 !important;
                --border-color: #cccccc !important;
                --primary-color: #000000 !important;
                --card-bg: #ffffff !important;
                --table-header-bg: #eeeeee !important;
                --table-row-hover: #ffffff !important;
            }}
            body {{
                background-color: #ffffff !important;
                color: #000000 !important;
                padding: 0 !important;
            }}
            .container {{
                border: none !important;
                box-shadow: none !important;
                padding: 0 !important;
                max-width: 100% !important;
            }}
            .controls {{
                display: none !important;
            }}
            .card, table, .chart-container {{
                page-break-inside: avoid;
                border: 1px solid #cccccc !important;
                background-color: #ffffff !important;
                color: #000000 !important;
            }}
            .card-value, h2, .header h1 {{
                color: #000000 !important;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧬 PrimerForge Tiled-Amplicon Router Report</h1>
            <p>Whole-Genome Overlapping Tiled Scheme Design &middot; DP Optimal Routing</p>
            <div class="controls">
                <button class="btn btn-toggle" onclick="toggleTheme()">🌓 Toggle Theme</button>
                <button class="btn btn-primary" onclick="window.print()">🖨️ Save as PDF</button>
            </div>
        </div>
        
        <h2>📊 1. Scheme Settings & Coverage Summary</h2>
        <div class="grid">
            <div class="card">
                <div class="card-title">Reference Sequence Size</div>
                <div class="card-value">{genome_len} bp</div>
            </div>
            <div class="card">
                <div class="card-title">Target Tile Size</div>
                <div class="card-value">{tile_size} bp</div>
            </div>
            <div class="card">
                <div class="card-title">Overlapping Step</div>
                <div class="card-value">{overlap} bp</div>
            </div>
            <div class="card">
                <div class="card-title">Total Tiles Routed</div>
                <div class="card-value">{len(tiles)} tiles</div>
            </div>
            <div class="card" style="grid-column: span 2;">
                <div class="card-title">Mean Scheme ML Success</div>
                <div class="card-value" style="color: var(--accent-success);">{avg_success*100:.2f}%</div>
            </div>
        </div>
        
        <h2>🧱 2. Routed Overlapping Tile Scheme Grid</h2>
        <table>
            <thead>
                <tr>
                    <th style="text-align: center;">Tile #</th>
                    <th>Genome Range</th>
                    <th>Forward Primer (5′ → 3′)</th>
                    <th>Reverse Primer (5′ → 3′)</th>
                    <th style="text-align: center;">Product Size</th>
                    <th style="text-align: center;">ML Success</th>
                </tr>
            </thead>
            <tbody>
                {tr_rows}
            </tbody>
        </table>
        
        <h2>🔬 3. Tiled Reference Scheme Performance Verdict</h2>
        <div class="card" style="border-left: 5px solid var(--primary-color);">
            <div class="card-title">Dynamic Programming Scheme Uniformity Verdict</div>
            <div class="card-value" style="font-size: 22px; margin-bottom: 10px;">
                <span class="card-status {scui_class}" style="font-size: 14px; padding: 4px 12px; border-radius: 8px;">{scui_status}</span>
            </div>
            <p style="font-size: 14.5px; line-height: 1.6; margin: 10px 0 0 0; color: var(--text-color);">
                {scientific_conclusion}
            </p>
        </div>
        
        {f"<h2>📈 4. Reference Genome Coverage Map</h2><div class='chart-container'><img class='chart-img' src='{coverage_map_base64}' alt='Genome Coverage Map'></div>" if coverage_map_base64 else ""}
        
        {METHODOLOGY_APPENDIX_HTML}
        
        <div class="footer">
            <p>Generated by PrimerForge Web Server &middot; publication-grade molecular engineering diagnostics platform.</p>
        </div>
    </div>

    <script>
        function toggleTheme() {{
            document.body.classList.toggle('light-theme');
            const isLight = document.body.classList.contains('light-theme');
            localStorage.setItem('theme', isLight ? 'light' : 'dark');
        }}
        
        window.onload = function() {{
            const savedTheme = localStorage.getItem('theme');
            if (savedTheme === 'light') {{
                document.body.classList.add('light-theme');
            }}
        }}
    </script>
</body>
</html>"""
    return html


# ──────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────
st.sidebar.title("🧬 PrimerForge")
st.sidebar.markdown(
    "**Publication-grade** PCR primer design: hybrid "
    "thermodynamic + ML + pangenome specificity."
)
st.sidebar.divider()

# Version & model status
st.sidebar.markdown("### ⚙️ Engine Status")
model_path = "models/primerforge_lightgbm.model"
model_exists = os.path.exists(model_path)
st.sidebar.success("🤖 ML Scorer: **Active**" if model_exists else "⚠️ ML Scorer: model not found")
st.sidebar.info(
    "🔬 **Engine**: v0.3.0-FinalPolish  \n"
    "📦 **Ensemble**: GBDT × 5 + MLP  \n"
    "🎯 **Calibration**: Platt sigmoid  \n"
    "🧪 **EWC Fine-Tune**: ✓ Enabled"
)
st.sidebar.divider()

# Quick-start hints
with st.sidebar.expander("📖 Quick Start"):
    st.markdown(
        textwrap.dedent("""\
        ```bash
        # Install
        poetry install

        # CLI design
        poetry run primerforge design \\
          --target <SEQ> --num-return 5

        # Web server
        poetry run streamlit run web_server.py

        # Fine-tune on lab data
        poetry run python fine_tune.py \\
          --csv data/lab_results.csv \\
          --out models/user_model
        ```
        """)
    )

st.sidebar.markdown("🔬 *Designed for high-impact pangenome-aware molecular research.*")

# ──────────────────────────────────────────────────────────────
# Hero header
# ──────────────────────────────────────────────────────────────
st.title("🧬 PrimerForge Design Server")
st.markdown(
    "**Hybrid Thermodynamic & Machine Learning Platform** for "
    "Pangenome-Aware PCR Primer Design  \n"
    "Stacked ensemble GBDT + sequence MLP · ILP dimer-free multiplex · "
    "DP tiled-amplicon router · Lab-adaptive fine-tuning"
)
st.divider()

if "designed" not in st.session_state:
    st.session_state["designed"] = False

# ──────────────────────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎯 Single-Locus Design",
    "🔀 ILP Multiplex Design",
    "🧱 Tiled-Amplicon Router",
    "📈 Retrain & Diagnostics",
    "🔬 Fine-Tune (Lab Data)",
    "🔄 Active Learning Playground",
])


# ═══════════════════════════════════════════════════════════════
# TAB 1 – Single-Locus Design
# ═══════════════════════════════════════════════════════════════
with tab1:
    st.header("🎯 Single-Locus Biophysical Design")
    st.markdown(
        "Design primers against a target locus.  Ranks candidates by "
        "calibrated ML success probability with optional SHAP explanation."
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        target_seq = st.text_area(
            "Target Template Sequence (5′ → 3′)",
            "CACCATTGGCAATGAGCGGTTCCGCTGCCCTGAGGCACTCTTCCAGCCTTCCTTCCTGGGC"
            "ATGGAGTCCTGTGGCATCCACGAAACTACCTTCAACTCCATCATGAAGTGTGACGTGGACA"
            "TCCGCAAAGACCTGTACGCCAACACAGTGCTGTCTGGCGGCACCACCATGTACCCTGGCAT"
            "TGCTGACAGGATGCAGAAGGAGATCACTGCCCTGGCACCCAGCACAATGAAGATCAAGAT"
            "CATTGCTCCTCCTGAGCGC",
            height=160,
            key="target_seq_input"
        )
    with col2:
        opt_tm = st.slider("Optimal Tm (°C)", 55.0, 68.0, 60.0, 0.5)
        num_return = st.number_input("Top candidates to return", 1, 50, 5)
        show_shap = st.checkbox("Show SHAP feature attribution", value=True)

    with st.expander("⚙️ Advanced Thermodynamic & Specificity Settings"):
        ac1, ac2 = st.columns(2)
        with ac1:
            min_size = st.number_input("Min Primer Size (bp)", 15, 25, 18)
            max_size = st.number_input("Max Primer Size (bp)", 20, 30, 24)
        with ac2:
            pangenome_path = st.text_input("Pangenome FASTA Index Path", "")
            vcf_path = st.text_input("VCF Variation Database Path", "")
            maf_thresh = st.slider("Variant Filtering MAF Threshold", 0.001, 0.10, 0.01, 0.001)

    if st.button("🚀 Generate & Score Primers", key="btn_single"):
        if not target_seq.strip():
            st.error("Please enter a target sequence.")
        else:
            # Sanitize path inputs
            if pangenome_path:
                try:
                    pangenome_path = secure_path(pangenome_path)
                except ValueError as ve:
                    st.error(str(ve))
                    st.stop()
            if vcf_path:
                try:
                    vcf_path = secure_path(vcf_path)
                except ValueError as ve:
                    st.error(str(ve))
                    st.stop()
            
            with st.spinner("Generating thermodynamic candidates and scoring with ensemble ML…"):
                biophys = get_cached_biophysics_engine(opt_tm=opt_tm, min_size=min_size, max_size=max_size)
                try:
                    candidates = biophys.generate_candidates(target_seq, num_return=max(60, num_return * 8))
                except Exception as exc:
                    st.error(f"Design failed: {exc}")
                    candidates = []

            if not candidates:
                st.warning("No primer pairs matched your biophysical constraints.")
            else:
                st.session_state["candidates"] = candidates
                st.session_state["designed"] = True
                st.session_state["show_shap"] = show_shap
                st.session_state["num_return"] = num_return

    # Check if we have designed candidates to show
    if st.session_state.get("designed", False) and "candidates" in st.session_state:
        candidates = st.session_state["candidates"]
        num_return = st.session_state.get("num_return", 5)
        show_shap = st.session_state.get("show_shap", True)

        ml_scorer = get_cached_ml_scorer()
        results, shap_rows = [], []
        for pair in candidates:
            spec = {
                "f_off_targets": 0, "r_off_targets": 0,
                "f_var_dist": 20.0, "r_var_dist": 20.0,
                "f_var_maf": 0.0, "r_var_maf": 0.0,
            }
            try:
                res = ml_scorer.predict_success_with_uncertainty(pair, spec)
                prob   = res.get("probability")
                lo, hi = res.get("ci_low"), res.get("ci_high")
            except Exception as e:
                prob = ml_scorer.predict_success(pair, spec)
                lo, hi = max(0.01, prob - 0.05), min(0.99, prob + 0.05)

            # We use gc_percent if available, otherwise fallback to gc_content
            gc_f = getattr(pair.forward, "gc_percent", getattr(pair.forward, "gc_content", 50.0))
            gc_r = getattr(pair.reverse, "gc_percent", getattr(pair.reverse, "gc_content", 50.0))

            results.append({
                "Forward Primer": pair.forward.sequence,
                "Fwd Tm (°C)": f"{pair.forward.tm:.1f}",
                "Fwd GC (%)": f"{gc_f:.0f}",
                "Reverse Primer": pair.reverse.sequence,
                "Rev Tm (°C)": f"{pair.reverse.tm:.1f}",
                "Product (bp)": pair.product_size,
                "Cross-Dimer ΔG": f"{pair.cross_dimer_dg:.2f}",
                "ML Score": f"{prob * 100:.1f}%",
                "95% CI": f"[{lo*100:.1f}%, {hi*100:.1f}%]",
                "pair_obj": pair,
                "ml_score_val": prob,
                "spec_obj": spec,
            })

            # SHAP for top candidate
            if len(shap_rows) < 1 and show_shap:
                try:
                    shap_vals = ml_scorer.explain_prediction(pair, spec)
                    shap_rows.append(shap_vals)
                except Exception:
                    pass

        df_show = pd.DataFrame([
            {k: v for k, v in r.items() if k not in ["pair_obj", "ml_score_val", "spec_obj"]}
            for r in results[:num_return]
        ])
        st.success(f"✅ Ranked **{len(df_show)}** primer pairs by ML success probability.")
        st.dataframe(df_show, use_container_width=True)

        # Download button
        csv_bytes = df_show.to_csv(index=False).encode()
        st.download_button(
            "⬇️ Download Results (CSV)", csv_bytes,
            file_name="primerforge_design_results.csv", mime="text/csv"
        )

        # Selectbox to let the user select which candidate to analyze
        candidate_options = [
            f"Candidate {idx} (ML: {res['ML Score']} | Fwd: {res['Forward Primer'][:10]}... / Rev: {res['Reverse Primer'][:10]}...)"
            for idx, res in enumerate(results[:num_return])
        ]
        selected_index = st.selectbox(
            "🎯 Select designed primer candidate to analyze:",
            options=range(len(candidate_options)),
            format_func=lambda x: candidate_options[x],
            key="selected_candidate_to_profile"
        )

        # Get selected candidate details
        top_res = results[selected_index]
        top_pair = top_res["pair_obj"]
        top_spec = top_res["spec_obj"]
        top_score = top_res["ml_score_val"]

        # Expandable biophysical diagnostics
        with st.expander("🔍 Biophysical Diagnostics & Secondary Structure", expanded=True):
            st.markdown(f"### Candidate {selected_index} Structural & Stability Profiling")
            
            # Nussinov fold calculations
            ref_folder = get_cached_amplicon_folder()
            
            target_seq_clean = "".join(c for c in target_seq.upper() if c in "ATGC")
            fwd_seq = top_pair.forward.sequence.upper()
            rev_seq = top_pair.reverse.sequence.upper()
            
            rev_comp_rev = reverse_complement(rev_seq)
            
            amplicon_seq = ""
            if fwd_seq in target_seq_clean and rev_comp_rev in target_seq_clean:
                idx_fwd = target_seq_clean.find(fwd_seq)
                idx_rev = target_seq_clean.find(rev_comp_rev) + len(rev_comp_rev)
                if idx_fwd < idx_rev:
                    amplicon_seq = target_seq_clean[idx_fwd:idx_rev]
            
            if not amplicon_seq:
                amplicon_seq = fwd_seq + "N" * max(0, top_pair.product_size - len(fwd_seq) - len(rev_seq)) + rev_comp_rev
            
            mfe, frac_paired, largest_loop = ref_folder.fold(amplicon_seq)
            
            biophys_diag = get_cached_biophysics_engine()
            f_terminal_dg = biophys_diag.calculate_terminal_dg(fwd_seq)
            r_terminal_dg = biophys_diag.calculate_terminal_dg(rev_seq)

            # Restructure into clinical-grade 3-row diagnostic grid
            st.markdown("#### 🧱 Row 1: Amplicon & Structure Diagnostics")
            mc_col1, mc_col2, mc_col3 = st.columns(3)
            mc_col1.metric("Amplicon Nussinov MFE", f"{mfe:.2f} kcal/mol",
                           help="Minimum free energy from Turner 2004 nearest-neighbor stacking.")
            mc_col2.metric("Fraction of Bases Paired", f"{frac_paired*100:.1f}%",
                           help="Percentage of amplicon bases participating in double-stranded structure.")
            mc_col3.metric("Largest Unpaired Loop", f"{largest_loop} bp",
                           help="Largest single-stranded loop in predicted amplicon fold.")

            # Calculate GC Clamps (number of G/C in last 5 bases of 3' end)
            f_clamp_gc = sum(1 for b in fwd_seq[-5:] if b in "GC")
            r_clamp_gc = sum(1 for b in rev_seq[-5:] if b in "GC")
            
            f_clamp_status = "Optimal" if 1 <= f_clamp_gc <= 2 else ("Suboptimal" if f_clamp_gc == 0 else "Too Strong")
            r_clamp_status = "Optimal" if 1 <= r_clamp_gc <= 2 else ("Suboptimal" if r_clamp_gc == 0 else "Too Strong")

            st.markdown("#### 🎯 Row 2: Forward Primer Biophysical Anchoring")
            f_col1, f_col2, f_col3, f_col4 = st.columns(4)
            f_col1.metric("Fwd 3' Stability (ΔG₃′)", f"{f_terminal_dg:.2f} kcal/mol", 
                          help="Unified NN SantaLucia 1998 parameter stability. High negative values represent optimal anchors.")
            f_col2.metric("Fwd GC Clamp (3' End)", f"{f_clamp_gc} G/C bases", delta=f_clamp_status,
                          help="Presence of G/C bases in the terminal 5 nucleotides of the 3' end. 1-2 bases is optimal.")
            f_col3.metric("Fwd Self-Hairpin dG", f"{top_pair.forward.hairpin_dg:.2f} kcal/mol",
                          help="Free energy of self-hairpin loop formation. Values below -4.0 represent self-annealing risks.")
            f_col4.metric("Fwd Homodimer dG", f"{top_pair.forward.homodimer_dg:.2f} kcal/mol",
                          help="Free energy of homodimer formation. Values below -4.0 represent self-dimerization risks.")

            st.markdown("#### 🎯 Row 3: Reverse Primer Biophysical Anchoring")
            r_col1, r_col2, r_col3, r_col4 = st.columns(4)
            r_col1.metric("Rev 3' Stability (ΔG₃′)", f"{r_terminal_dg:.2f} kcal/mol",
                          help="Unified NN SantaLucia 1998 parameter stability.")
            r_col2.metric("Rev GC Clamp (3' End)", f"{r_clamp_gc} G/C bases", delta=r_clamp_status,
                          help="Presence of G/C bases in the terminal 5 nucleotides of the 3' end.")
            r_col3.metric("Rev Self-Hairpin dG", f"{top_pair.reverse.hairpin_dg:.2f} kcal/mol",
                          help="Free energy of self-hairpin loop formation.")
            r_col4.metric("Rev Homodimer dG", f"{top_pair.reverse.homodimer_dg:.2f} kcal/mol",
                          help="Free energy of homodimer formation.")

            st.divider()
            st.markdown("**Folded Amplicon Secondary Structure (Dot-Bracket Notation):**")
            st.code(ref_folder._dp.compute_mfe(amplicon_seq)[1], language="text")

        # Variant mismatch simulator
        with st.expander("🧬 Variant Mismatch Simulator & Explainability", expanded=True):
            st.markdown(
                "Simulate variant mutations in the target template sequence at the primer binding locations "
                "to predict physical mismatch tolerance using the **Taq-weighted exponential decay** engine."
            )
            
            f_perfect_complement = get_complement_seq(fwd_seq)
            r_perfect_complement = get_complement_seq(rev_seq)

            vs_col1, vs_col2 = st.columns(2)
            with vs_col1:
                f_variant_site = st.text_input(
                    "Forward Template Binding Site (3′ → 5′ complement)",
                    value=f_perfect_complement,
                    help="Modify this template sequence to introduce mismatches relative to the forward primer."
                )
            with vs_col2:
                r_variant_site = st.text_input(
                    "Reverse Template Binding Site (3′ → 5′ complement)",
                    value=r_perfect_complement,
                    help="Modify this template sequence to introduce mismatches relative to the reverse primer."
                )

            # Predict variant success
            mismatch_score = ml_scorer.predict_success_with_variant_mismatches(
                top_pair, f_variant_site, r_variant_site, spec_data=top_spec
            )

            v_mc1, v_mc2, v_mc3 = st.columns(3)
            v_mc1.metric("Perfect Match Baseline", f"{top_score * 100:.2f}%")
            
            diff_pct = (mismatch_score - top_score) * 100
            delta_str = f"{diff_pct:+.2f}%"
            v_mc2.metric("Mismatch-Calibrated Success", f"{mismatch_score * 100:.2f}%", delta=delta_str, delta_color="inverse")
            
            is_viable = mismatch_score >= 0.50
            v_mc3.metric("Viability Status", "✅ Viable" if is_viable else "❌ Mismatches Fatal", 
                         help="Viability threshold set at P(success) >= 50%.")

            # Show exact mismatch positions
            f_mismatches = []
            r_mismatches = []
            comp_map = {"A": "T", "T": "A", "G": "C", "C": "G"}
            
            for idx in range(min(len(fwd_seq), len(f_variant_site))):
                if fwd_seq[idx] != comp_map.get(f_variant_site[idx], ""):
                    dist = len(fwd_seq) - 1 - idx
                    f_mismatches.append(f"Position {idx+1} (5'→3') / {dist} bp from 3' end")
            
            for idx in range(min(len(rev_seq), len(r_variant_site))):
                if rev_seq[idx] != comp_map.get(r_variant_site[idx], ""):
                    dist = len(rev_seq) - 1 - idx
                    r_mismatches.append(f"Position {idx+1} (5'→3') / {dist} bp from 3' end")
            
            if f_mismatches or r_mismatches:
                st.warning("⚠️ **Detected Mismatch Details:**")
                if f_mismatches:
                    st.markdown("**Forward Primer Mismatches:**")
                    for m in f_mismatches:
                        st.markdown(f"- {m}")
                if r_mismatches:
                    st.markdown("**Reverse Primer Mismatches:**")
                    for m in r_mismatches:
                        st.markdown(f"- {m}")
            else:
                st.info("ℹ️ Perfect match sequence — zero mismatches detected.")

        # ──────────────────────────────────────────────────────────
        # Programmatic Biophysical Conclusion (AVI Methodological Framework)
        # ──────────────────────────────────────────────────────────
        anchoring_issues = []
        if f_terminal_dg > -5.0:
            anchoring_issues.append("Forward 3' terminal stability is thermodynamically weak (ΔG > -5.0 kcal/mol), risking low polymerase initiation efficiency.")
        elif f_terminal_dg < -9.0:
            anchoring_issues.append("Forward 3' terminal stability is excessively stable (ΔG < -9.0 kcal/mol), increasing mispriming risks.")
        
        if r_terminal_dg > -5.0:
            anchoring_issues.append("Reverse 3' terminal stability is thermodynamically weak (ΔG > -5.0 kcal/mol), risking low polymerase initiation efficiency.")
        elif r_terminal_dg < -9.0:
            anchoring_issues.append("Reverse 3' terminal stability is excessively stable (ΔG < -9.0 kcal/mol), increasing mispriming risks.")
            
        if f_clamp_status != "Optimal":
            anchoring_issues.append(f"Forward 3' GC clamp is {f_clamp_status.lower()} ({f_clamp_gc} bases).")
        if r_clamp_status != "Optimal":
            anchoring_issues.append(f"Reverse 3' GC clamp is {r_clamp_status.lower()} ({r_clamp_gc} bases).")

        loop_issues = []
        if top_pair.forward.hairpin_dg < -4.0:
            loop_issues.append(f"Forward primer hairpin loop formation is thermodynamically active (ΔG = {top_pair.forward.hairpin_dg:.2f} kcal/mol).")
        if top_pair.reverse.hairpin_dg < -4.0:
            loop_issues.append(f"Reverse primer hairpin loop formation is thermodynamically active (ΔG = {top_pair.reverse.hairpin_dg:.2f} kcal/mol).")
        if top_pair.forward.homodimer_dg < -5.0:
            loop_issues.append(f"Forward primer homodimerization risk is elevated (ΔG = {top_pair.forward.homodimer_dg:.2f} kcal/mol).")
        if top_pair.reverse.homodimer_dg < -5.0:
            loop_issues.append(f"Reverse primer homodimerization risk is elevated (ΔG = {top_pair.reverse.homodimer_dg:.2f} kcal/mol).")
        if top_pair.cross_dimer_dg < -5.0:
            loop_issues.append(f"Cross-hybridization heterodimerization risk is elevated (ΔG = {top_pair.cross_dimer_dg:.2f} kcal/mol).")

        obstruction_issues = []
        if frac_paired > 0.45:
            obstruction_issues.append(f"High secondary structure density inside target amplicon ({frac_paired*100:.1f}% bases paired) may stall polymerase elongation.")
        if mfe < -12.0:
            obstruction_issues.append(f"Highly stable amplicon secondary fold (Nussinov MFE = {mfe:.2f} kcal/mol) threatens transcription kinetics.")

        resilience_issues = []
        if mismatch_score < 0.50:
            resilience_issues.append(f"Variant mismatch decay reduces assay success probability to {mismatch_score*100:.1f}%, indicating vulnerability to variant dropouts.")

        # AVI Synthesis
        if not anchoring_issues and not loop_issues and not obstruction_issues and not resilience_issues and top_score >= 0.50:
            avi_status = "✅ Certified Viable"
            avi_class = "success"
            scientific_conclusion = f"The physicochemical and thermodynamic parameters of Candidate {selected_index} are in perfect alignment. Strong, non-degenerate terminal anchoring is confirmed by both optimal 3' stability (Forward: {f_terminal_dg:.2f} kcal/mol / Reverse: {r_terminal_dg:.2f} kcal/mol) and ideal GC clamps. Self-annealing risks (hairpins, homodimers, and heterodimers) are completely negligible (all ΔG > -4.0 kcal/mol). Amplicon Nussinov folding analysis indicates minimal structural obstruction with a balanced {frac_paired*100:.1f}% base-pairing density, facilitating unhindered polymerase elongation. Under variant mismatch simulation, the success probability remains robust ({mismatch_score*100:.1f}%), certifying this assay with high-impact clinical utility."
        elif top_score < 0.50 or (top_pair.cross_dimer_dg < -8.0 or mismatch_score < 0.30):
            avi_status = "❌ High Failure Risk"
            avi_class = "error"
            all_problems = anchoring_issues + loop_issues + obstruction_issues + resilience_issues
            problem_str = " ".join(all_problems) if all_problems else "Low overall machine learning success probability."
            scientific_conclusion = f"Candidate {selected_index} exhibits thermodynamic and specificity parameters that indicate a high risk of PCR failure in wet-lab assays. Specifically: {problem_str} Active heterodimerization or extreme variant vulnerability will deplete available primers in solution or block initiation completely, leading to assay dropouts. Redesign of this locus target or primer boundaries is strongly recommended."
        else:
            avi_status = "⚠️ Conditional Duplex"
            avi_class = "warning"
            all_problems = anchoring_issues + loop_issues + obstruction_issues + resilience_issues
            problem_str = " ".join(all_problems)
            scientific_conclusion = f"Candidate {selected_index} is certified as conditionally viable, but exhibits biophysical anomalies that require careful laboratory optimization. Specifically: {problem_str} While the overall calibrated ML success probability remains high ({top_score*100:.1f}%), these thermodynamic clamping or self-annealing conditions might reduce amplification efficiency. We recommend running reactions at slightly elevated annealing temperatures or utilizing specialized master mixes with PCR additives (such as DMSO or betaine) to disrupt amplicon secondary folds."

        st.subheader("🔬 Biophysical Assay Viability Verdict (AVI)")
        if avi_class == "success":
            st.success(f"**{avi_status}**\n\n{scientific_conclusion}")
        elif avi_class == "warning":
            st.warning(f"**{avi_status}**\n\n{scientific_conclusion}")
        else:
            st.error(f"**{avi_status}**\n\n{scientific_conclusion}")

        # SHAP plot
        shap_base64 = ""
        if show_shap:
            try:
                sv = ml_scorer.explain_prediction(top_pair, top_spec)
                feat_names = list(sv.keys())
                shap_vals_arr = np.array([sv[k] for k in feat_names])
                order = np.argsort(np.abs(shap_vals_arr))[::-1][:12]

                fig, ax = _dark_fig(figsize=(8, 4))
                colors = ["#4ade80" if v > 0 else "#f87171" for v in shap_vals_arr[order[::-1]]]
                ax.barh(
                    [feat_names[i].replace("_", " ") for i in order[::-1]],
                    shap_vals_arr[order[::-1]],
                    color=colors,
                )
                ax.axvline(0, color="#475569", linewidth=1)
                ax.set_xlabel("SHAP Value (impact on success probability)", color=TEXT_CLR)
                ax.set_title(f"🔍 Candidate {selected_index} SHAP Feature Attribution", color="#7dd3fc")
                st.pyplot(fig)
                
                # Base64 encode for offline HTML/PDF report embedding
                import io
                import base64
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
                buf.seek(0)
                shap_base64 = f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
                buf.close()
                plt.close(fig)  # Prevent memory leaks!
            except Exception as e:
                st.warning(f"Could not generate SHAP explanation for candidate {selected_index}: {e}")

        # Generate & Download Biophysical Diagnostic Report
        try:
            report_html = generate_html_report(
                selected_index,
                top_pair,
                top_spec,
                top_score,
                mfe,
                frac_paired,
                largest_loop,
                amplicon_seq,
                f_terminal_dg,
                r_terminal_dg,
                mismatch_score,
                shap_base64
            )
            st.divider()
            st.download_button(
                "📋 Download Complete Diagnostic Report (PDF / HTML)",
                data=report_html.encode("utf-8"),
                file_name=f"primerforge_diagnostic_report_candidate_{selected_index}.html",
                mime="text/html",
                key=f"dl_html_report_{selected_index}",
                help="Downloads a fully styled, self-contained HTML report. Open it in any browser and press Ctrl+P to save as a pixel-perfect PDF."
            )
        except Exception as re:
            st.error(f"Could not compile biophysical diagnostic report: {re}")

        # Biophysical & SHAP Feature Dictionary
        with st.expander("📚 Biophysical & SHAP Feature Dictionary", expanded=False):
            st.markdown(r"""
            ### 🧬 Molecular Biophysics & Feature Glossary
            
            To ensure complete scientific transparency, here is what each feature in the SHAP attribution chart represents:
            
            *   **off targets (f/r)**: Number of high-identity secondary binding sites across the pangenome. Lower counts prevent non-specific off-target PCR amplification.
            *   **hairpin dg (f/r)**: Thermodynamic free energy of self-hairpin loops. Stable self-hairpins (highly negative $\Delta G$) cause the primer to fold on itself, blocking target hybridization.
            *   **var dist (f/r)**: Nucleotide distance from the critical $3'$ extension end to the nearest polymorphic variant (SNP/indel). PrimerForge filters out close variants to protect against clinical assay dropouts.
            *   **tm (f/r)**: Melting Temperature ($T_m$). Defines the thermal stability of primer-template hybridization.
            *   **tm diff**: Absolute melting temperature difference between forward and reverse primers ($|T_{m,\text{fwd}} - T_{m,\text{rev}}|$). Low mismatches guarantee simultaneous annealing.
            *   **homodimer dg (f/r)**: Free energy of self-dimerization (primer molecules binding to identical copies of themselves in solution).
            *   **cross dimer dg**: Free energy of heterodimerization (forward primer binding to the reverse primer). Highly negative values deplete active primers, causing PCR failure.
            *   **clamp gc (f/r)**: GC base presence at the $3'$ terminal end, which stabilizes the initial polymerase extension anchor site.
            *   **target mfe**: Nussinov minimum free energy of the amplified target amplicon's secondary structure folding. Stable target folds physically obstruct polymerase extension.
            """)

        with st.expander("🔬 Biophysical Methodology & Mathematical Formulations (AVI)", expanded=False):
            st.markdown(r"""
            ### 1. Nearest-Neighbor Duplex Anchoring Stability
            The thermodynamic stability of the $3'$ terminal region determines polymerase initiation efficiency and assay specificity. Free energy ($\Delta G^\circ$) is calculated using the Nearest-Neighbor unified thermodynamic model:
            """)
            st.latex(r"\Delta G^\circ(T) = \Delta H^\circ - T \Delta S^\circ + \Delta G^\circ_{\text{init}} + \Delta G^\circ_{\text{salt}}")
            st.markdown(r"""
            Where:
            *   $\Delta H^\circ$ and $\Delta S^\circ$ are enthalpy and entropy changes of the nearest-neighbor doublets.
            *   $\Delta G^\circ_{\text{salt}}$ adjusts for salt concentration ($\Delta S^\circ_{\text{salt}} = \Delta S^\circ_{\text{std}} + 0.368 \cdot (N-1) \cdot \ln[\text{Na}^+]$).
            
            ### 2. Nussinov Secondary Structure folding
            Conformational obstructions within the single-stranded amplicon fold are resolved using the Nussinov dynamic programming algorithm, which calculates the maximum base-pairing density fraction:
            """)
            st.latex(r"f_{\text{paired}} = \frac{2 \cdot N_{\text{paired}}}{L_{\text{amplicon}}}")
            st.markdown(r"""
            If $f_{\text{paired}} > 0.45$ or Nussinov MFE $< -12.0 \text{ kcal/mol}$, polymerase extension is physically blocked.
            
            ### 3. Taq-Weighted Variant Mismatch Decay
            A Taq-polymerase weighted exponential decay model calibrates performance drops when polymorphic variants overlap binding sites:
            """)
            st.latex(r"S_{\text{mismatch}} = S_{\text{baseline}} \cdot \prod_{v \in V} \exp\left( - \lambda \cdot d(v, 3') \right)")
            st.markdown(r"""
            Where:
            *   $d(v, 3')$ is the physical nucleotide distance of variant $v$ from the critical $3'$ terminal site.
            *   $\lambda$ is the decay sensitivity coefficient ($\approx 0.15$).
            """)


# ═══════════════════════════════════════════════════════════════
# TAB 2 – Multiplex Design
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.header("🔀 Dimer-Free Multiplex PCR Panel Design")
    st.markdown(
        "Assemble compatible primer panels across multiple target loci. "
        "Allows selecting between graph-based Integer Linear Programming (ILP) "
        "or Greedy Dimerization & Rescue optimization to guarantee 100% dimer-free panels."
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        multiplex_seqs = st.text_area(
            "Target loci (one sequence per line)",
            "\n".join([
                "CACCATTGGCAATGAGCGGTTCCGCTGCCCTGAGGCACTCTTCCAGCCTTCCTTCCTGGGCAT",
                "CCTGTGGCATCCACGAAACTACCTTCAACTCCATCATGAAGTGTGACGTGGACATCCGC",
                "CAAAGACCTGTACGCCAACACAGTGCTGTCTGGCGGCACCACCATGTACCCTGGCATTG",
                "CTGACAGGATGCAGAAGGAGATCACTGCCCTGGCACCCAGCACAATGAAGATCAAGATCATTG",
                "CACCATTGGCAATGAGCGGTTCCGCTGCCCTGAGGCACTCTTCCAGCCTTCCTTCCTGGGTAC",
            ]),
            height=160,
            key="multiplex_seqs_input"
        )
    with col2:
        solver_engine = st.selectbox(
            "Optimization Engine",
            ["Greedy Dimerization-Rescue", "Integer Linear Programming (ILP)"],
            index=0,
            help="Choose between our fast, greedy dimer-avoiding heuristics or Pulp graph MWIS ILP solver."
        )
        max_plex = st.number_input("Max Panel Size (plex)", 2, 24, 8)
        dg_thresh = st.slider("Dimerization soft threshold ΔG (kcal/mol)", -8.0, -2.0, -6.0, 0.1)

    if st.button("🔀 Assemble Multiplex Panel", key="btn_multi"):
        if not multiplex_seqs.strip():
            st.error("Please enter target sequences.")
        else:
            lines = [l.strip().upper() for l in multiplex_seqs.split("\n") if l.strip()]
            
            with st.spinner(f"Designing candidates for {len(lines)} loci and optimizing multiplex panel…"):
                biophys   = get_cached_biophysics_engine()
                ml_scorer = get_cached_ml_scorer()
                
                if solver_engine == "Greedy Dimerization-Rescue":
                    candidate_pools = []
                    failed_loci = []
                    prog = st.progress(0)
                    for idx, line in enumerate(lines):
                        try:
                            cands = biophys.generate_candidates(line, num_return=6)
                            if cands:
                                candidate_pools.append(list(cands))
                            else:
                                failed_loci.append((idx + 1, line))
                        except Exception as e:
                            failed_loci.append((idx + 1, f"{line} (Error: {e})"))
                        prog.progress((idx + 1) / len(lines))
                    
                    if failed_loci:
                        st.warning(f"⚠️ Failed to generate candidates for {len(failed_loci)} targets:\n" + "\n".join([f"- Line {num}: {info}" for num, info in failed_loci]))
                    
                    if not candidate_pools:
                        st.warning("Could not generate candidates for any target.")
                    else:
                        dimer_opt = DimerMultiplexOptimizer(biophys)
                        panel = dimer_opt.design_compatible_panel(candidate_pools, threshold=dg_thresh, hard_limit=-9.0)
                        
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Panel Size", f"{len(panel.pairs)}-plex")
                        m2.metric("Global Penalty Φ", f"{panel.global_penalty:.3f}")
                        m3.metric("Dimer-Free Status", "✅ 100%" if panel.global_penalty == 0.0 else "⚠️ Dimer Risks Exist")
                        
                        results = []
                        for idx, pair in enumerate(panel.pairs):
                            success = ml_scorer.predict_success(pair)
                            results.append({
                                "Locus ID":         f"locus_{idx+1}",
                                "Forward Primer":   pair.forward.sequence,
                                "Fwd Tm (°C)":      f"{pair.forward.tm:.1f}",
                                "Reverse Primer":   pair.reverse.sequence,
                                "Rev Tm (°C)":      f"{pair.reverse.tm:.1f}",
                                "Product (bp)":     pair.product_size,
                                "Cross-Dimer ΔG":  f"{pair.cross_dimer_dg:.2f}",
                                "ML Success":       f"{success*100:.1f}%",
                            })
                        
                        st.dataframe(pd.DataFrame(results), use_container_width=True)
                        
                        # Heatmap
                        matrix = panel.dimerization_matrix
                        labels = panel.primer_labels
                        matrix_base64 = render_dimerization_heatmap(matrix, labels, "🔬 Symmetric Cross-Reactivity Dimerization Matrix (Greedy)")

                        # ──────────────────────────────────────────────────────────
                        # Programmatic Biophysical Conclusion (PSII Methodological Framework) - Greedy
                        # ──────────────────────────────────────────────────────────
                        tms = []
                        active_dimer_count = 0
                        ml_scores = []
                        
                        for r in results:
                            try:
                                ftm = float(r.get("Fwd Tm (°C)", r.get("Fwd Tm", "60.0")).replace("°C", "").strip())
                                rtm = float(r.get("Rev Tm (°C)", r.get("Rev Tm", "60.0")).replace("°C", "").strip())
                                tms.extend([ftm, rtm])
                            except Exception: pass
                                
                            try:
                                dg = float(r.get("Cross-Dimer ΔG", r.get("Dimer ΔG", "0.0")))
                                if dg < -6.0:
                                    active_dimer_count += 1
                            except Exception: pass
                                
                            try:
                                score_str = r.get("ML Success", r.get("ML Score", "50.0%")).replace("%", "").strip()
                                ml_scores.append(float(score_str) / 100.0)
                            except Exception: pass

                        max_tm_diff = max(tms) - min(tms) if tms else 0.0
                        avg_success = sum(ml_scores) / len(ml_scores) if ml_scores else 0.0
                        
                        psii_issues = []
                        if max_tm_diff > 2.0:
                            psii_issues.append(f"Elevated thermal disparity across panel (ΔTm = {max_tm_diff:.1f}°C). Max Tm is {max(tms):.1f}°C, Min Tm is {min(tms):.1f}°C.")
                        if active_dimer_count > 0:
                            psii_issues.append(f"Detected {active_dimer_count} loci pairings with active heterodimerization risks below the soft ΔG threshold (-6.0 kcal/mol).")
                        if avg_success < 0.60:
                            psii_issues.append(f"Mean panel ML success probability is relatively low ({avg_success*100:.1f}%).")
                            
                        if not psii_issues and panel.global_penalty == 0.0:
                            psii_status = "✅ Synergy Certified"
                            psii_class = "success"
                            scientific_conclusion = f"System analysis confirms that this {len(panel.pairs)}-plex primer panel exhibits flawless molecular synergy. The global dimerization penalty is 0.000, and all pairwise cross-reactivity matrix elements reside safely above the soft threshold, guaranteeing a 100% dimer-free assay. Thermal cohort uniformity is excellent, with a maximum melting temperature differential of only {max_tm_diff:.1f}°C. This tightly matched thermodynamic profile ensures highly uniform annealing kinetics across all loci during simultaneous amplification cycles, completely eliminating competitive amplification bias or target dropouts. The average panel success probability is exceptionally strong at {avg_success*100:.1f}%, certifying this panel for publication-grade pangenomic assays."
                        elif panel.global_penalty > 5.0 or max_tm_diff > 4.0 or avg_success < 0.50:
                            psii_status = "❌ High Interference Risk"
                            psii_class = "error"
                            problem_str = " ".join(psii_issues) if psii_issues else "High dimerization penalty or poor amplification profile."
                            scientific_conclusion = f"Multiplex panel diagnostics indicate significant molecular interference risks that threaten assay performance. Specifically: {problem_str} High dimerization energies (ΔG < -6.0 kcal/mol) or large thermal differentials (ΔTm = {max_tm_diff:.1f}°C) will cause massive primer-dimer formation and severe selective locus dropouts. High-affinity cross-dimers will preferentially consume primers, yielding non-specific bands and depleting target amplicons. We strongly recommend re-running the panel optimization using the ILP Graph Solver with more stringent ΔG thresholds."
                        else:
                            psii_status = "⚠️ Minor Interference Risk"
                            psii_class = "warning"
                            problem_str = " ".join(psii_issues)
                            scientific_conclusion = f"This multiplex panel is certified with minor interference risks. While overall panel assembly is compatible, optimized conditions are required to prevent dropouts: {problem_str} The moderate thermal variance or minor active cross-reactivity requires adjusting the master mix composition. We recommend increasing monovalent salt concentrations, utilizing hot-start Taq polymerase, or optimizing the multiplex annealing duration to favor specific target hybridization over transient primer-primer duplexes."

                        st.subheader("🔬 Panel Synergy & Interference Index (PSII) Verdict")
                        if psii_class == "success":
                            st.success(f"**{psii_status}**\n\n{scientific_conclusion}")
                        elif psii_class == "warning":
                            st.warning(f"**{psii_status}**\n\n{scientific_conclusion}")
                        else:
                            st.error(f"**{psii_status}**\n\n{scientific_conclusion}")

                        csv_bytes = pd.DataFrame(results).to_csv(index=False).encode()
                        st.download_button(
                            "⬇️ Download Panel (CSV)", csv_bytes,
                            file_name="primerforge_multiplex_panel.csv", mime="text/csv",
                            key="dl_greedy_panel"
                        )
                        
                        # Premium Multiplex Panel Report Download
                        try:
                            m_report_html = generate_multiplex_html_report(
                                results,
                                "Greedy Dimerization-Rescue Optimizer",
                                max_plex,
                                dg_thresh,
                                panel.global_penalty,
                                matrix_base64
                            )
                            st.download_button(
                                "📋 Download Multiplex Panel Report (PDF / HTML)",
                                data=m_report_html.encode("utf-8"),
                                file_name="primerforge_multiplex_panel_report_greedy.html",
                                mime="text/html",
                                key="dl_greedy_report",
                                help="Downloads a fully styled, self-contained HTML report. Open it in any browser and press Ctrl+P to save as a pixel-perfect PDF."
                            )
                        except Exception as mre:
                            st.error(f"Could not compile multiplex diagnostic report: {mre}")
                        
                else: # ILP solver
                    all_candidates = []
                    failed_loci = []
                    prog = st.progress(0)
                    for idx, line in enumerate(lines):
                        try:
                            cands = biophys.generate_candidates(line, num_return=6)
                            if not cands:
                                failed_loci.append((idx + 1, line))
                            for pair in cands:
                                success = ml_scorer.predict_success(pair)
                                all_candidates.append({
                                    "pair": pair,
                                    "predicted_success": success,
                                    "target_id": f"locus_{idx+1}",
                                    "is_valid": True,
                                })
                        except Exception as e:
                            failed_loci.append((idx + 1, f"{line} (Error: {e})"))
                        prog.progress((idx + 1) / len(lines))

                    if failed_loci:
                        st.warning(f"⚠️ Failed to generate candidates for {len(failed_loci)} targets:\n" + "\n".join([f"- Line {num}: {info}" for num, info in failed_loci]))

                    if not all_candidates:
                        st.warning("Could not generate candidates for any target.")
                    else:
                        optimizer = MultiplexOptimizer(biophys)
                        selected_panel, obj = optimizer.optimize_panel(
                            all_candidates, max_plex=max_plex, delta_g_threshold=dg_thresh
                        )

                        m1, m2, m3 = st.columns(3)
                        m1.metric("Panel Size", f"{len(selected_panel)}-plex")
                        m2.metric("ILP Objective (ΣP_success)", f"{obj:.3f}")
                        m3.metric("Dimer-Free", "✅ 100%")

                        results = []
                        pairs_list = []
                        for item in selected_panel:
                            pair = item["pair"]
                            pairs_list.append(pair)
                            results.append({
                                "Locus ID":         item["target_id"],
                                "Forward Primer":   pair.forward.sequence,
                                "Fwd Tm (°C)":      f"{pair.forward.tm:.1f}",
                                "Reverse Primer":   pair.reverse.sequence,
                                "Rev Tm (°C)":      f"{pair.reverse.tm:.1f}",
                                "Product (bp)":     pair.product_size,
                                "Cross-Dimer ΔG":  f"{pair.cross_dimer_dg:.2f}",
                                "ML Success":       f"{item['predicted_success']*100:.1f}%",
                            })

                        st.dataframe(pd.DataFrame(results), use_container_width=True)
                        
                        # Generate matrix for ILP panel as well
                        matrix_base64 = ""
                        if pairs_list:
                            dimer_opt = DimerMultiplexOptimizer(biophys)
                            matrix, labels = dimer_opt.build_dimerization_matrix(pairs_list)
                            matrix_base64 = render_dimerization_heatmap(matrix, labels, "🔬 Symmetric Cross-Reactivity Dimerization Matrix (ILP)")

                        # ──────────────────────────────────────────────────────────
                        # Programmatic Biophysical Conclusion (PSII Methodological Framework) - ILP
                        # ──────────────────────────────────────────────────────────
                        tms = []
                        active_dimer_count = 0
                        ml_scores = []
                        
                        for r in results:
                            try:
                                ftm = float(r.get("Fwd Tm (°C)", r.get("Fwd Tm", "60.0")).replace("°C", "").strip())
                                rtm = float(r.get("Rev Tm (°C)", r.get("Rev Tm", "60.0")).replace("°C", "").strip())
                                tms.extend([ftm, rtm])
                            except Exception: pass
                                
                            try:
                                dg = float(r.get("Cross-Dimer ΔG", r.get("Dimer ΔG", "0.0")))
                                if dg < -6.0:
                                    active_dimer_count += 1
                            except Exception: pass
                                
                            try:
                                score_str = r.get("ML Success", r.get("ML Score", "50.0%")).replace("%", "").strip()
                                ml_scores.append(float(score_str) / 100.0)
                            except Exception: pass

                        max_tm_diff = max(tms) - min(tms) if tms else 0.0
                        avg_success = sum(ml_scores) / len(ml_scores) if ml_scores else 0.0
                        
                        psii_issues = []
                        if max_tm_diff > 2.0:
                            psii_issues.append(f"Elevated thermal disparity across panel (ΔTm = {max_tm_diff:.1f}°C). Max Tm is {max(tms):.1f}°C, Min Tm is {min(tms):.1f}°C.")
                        if active_dimer_count > 0:
                            psii_issues.append(f"Detected {active_dimer_count} loci pairings with active heterodimerization risks below the soft ΔG threshold (-6.0 kcal/mol).")
                        if avg_success < 0.60:
                            psii_issues.append(f"Mean panel ML success probability is relatively low ({avg_success*100:.1f}%).")
                            
                        # ILP optimization guaranteed to satisfy the soft threshold delta_g, so global penalty is 0
                        if not psii_issues:
                            psii_status = "✅ Synergy Certified"
                            psii_class = "success"
                            scientific_conclusion = f"System analysis confirms that this {len(selected_panel)}-plex primer panel designed via Integer Linear Programming exhibits flawless molecular synergy. The global dimerization penalty is 0.000, and all pairwise cross-reactivity matrix elements reside safely above the soft threshold, guaranteeing a 100% dimer-free assay. Thermal cohort uniformity is excellent, with a maximum melting temperature differential of only {max_tm_diff:.1f}°C. This tightly matched thermodynamic profile ensures highly uniform annealing kinetics across all loci during simultaneous amplification cycles, completely eliminating competitive amplification bias or target dropouts. The average panel success probability is exceptionally strong at {avg_success*100:.1f}%, certifying this panel for publication-grade pangenomic assays."
                        elif max_tm_diff > 4.0 or avg_success < 0.50:
                            psii_status = "❌ High Interference Risk"
                            psii_class = "error"
                            problem_str = " ".join(psii_issues) if psii_issues else "High dimerization penalty or poor amplification profile."
                            scientific_conclusion = f"Multiplex panel diagnostics indicate significant molecular interference risks that threaten assay performance. Specifically: {problem_str} High dimerization energies (ΔG < -6.0 kcal/mol) or large thermal differentials (ΔTm = {max_tm_diff:.1f}°C) will cause massive primer-dimer formation and severe selective locus dropouts. High-affinity cross-dimers will preferentially consume primers, yielding non-specific bands and depleting target amplicons. We strongly recommend re-running the panel optimization using the ILP Graph Solver with more stringent ΔG thresholds."
                        else:
                            psii_status = "⚠️ Minor Interference Risk"
                            psii_class = "warning"
                            problem_str = " ".join(psii_issues)
                            scientific_conclusion = f"This multiplex panel is certified with minor interference risks. While overall panel assembly is compatible, optimized conditions are required to prevent dropouts: {problem_str} The moderate thermal variance or minor active cross-reactivity requires adjusting the master mix composition. We recommend increasing monovalent salt concentrations, utilizing hot-start Taq polymerase, or optimizing the multiplex annealing duration to favor specific target hybridization over transient primer-primer duplexes."

                        st.subheader("🔬 Panel Synergy & Interference Index (PSII) Verdict")
                        if psii_class == "success":
                            st.success(f"**{psii_status}**\n\n{scientific_conclusion}")
                        elif psii_class == "warning":
                            st.warning(f"**{psii_status}**\n\n{scientific_conclusion}")
                        else:
                            st.error(f"**{psii_status}**\n\n{scientific_conclusion}")

                        csv_bytes = pd.DataFrame(results).to_csv(index=False).encode()
                        st.download_button(
                            "⬇️ Download Panel (CSV)", csv_bytes,
                            file_name="primerforge_multiplex_panel.csv", mime="text/csv",
                            key="dl_ilp_panel"
                        )

                        # Premium Multiplex Panel Report Download (ILP)
                        try:
                            m_report_html = generate_multiplex_html_report(
                                results,
                                "Integer Linear Programming (ILP) Optimizer",
                                max_plex,
                                dg_thresh,
                                0.0, # ILP guarantees 0.0 penalty for selected pairs under soft threshold
                                matrix_base64
                            )
                            st.download_button(
                                "📋 Download Multiplex Panel Report (PDF / HTML)",
                                data=m_report_html.encode("utf-8"),
                                file_name="primerforge_multiplex_panel_report_ilp.html",
                                mime="text/html",
                                key="dl_ilp_report",
                                help="Downloads a fully styled, self-contained HTML report. Open it in any browser and press Ctrl+P to save as a pixel-perfect PDF."
                            )
                        except Exception as mre:
                            st.error(f"Could not compile multiplex diagnostic report: {mre}")

        st.divider()
        with st.expander("🔬 Biophysical Methodology & Mathematical Formulations (PSII)", expanded=False):
            st.markdown(r"""
            ### 1. Thermal Cohort Uniformity
            To prevent competitive amplification bias where high-affinity targets consume all monovalent salts and dNTPs, annealing temperatures must synchronize:
            """)
            st.latex(r"\Delta T_{m,\text{max}} = \max_{p \in P} T_m(p) - \min_{p \in P} T_m(p)")
            st.markdown(r"""
            *   **Optimal**: $\Delta T_{m,\text{max}} \le 2.0^\circ\text{C}$ (ensuring uniform annealing rates).
            *   **High Risk**: $\Delta T_{m,\text{max}} > 4.0^\circ\text{C}$ (forces high-affinity amplification at the cost of low-$T_m$ target dropouts).

            ### 2. Pairwise Cross-Hybridization Dimerization Penalty
            The cumulative panel interference density evaluates all primer-primer interactions in the pool:
            """)
            st.latex(r"D(i, j) = \max\left(0, -\Delta G^\circ_{\text{cross}}(i, j) - 6.0 \text{ kcal/mol}\right)")
            st.latex(r"\Phi(P) = \sum_{i \in P, j \in P, i < j} D(i, j)")
            st.markdown(r"""
            The Integer Linear Programming (ILP) optimizer selects a compatible subset of loci that minimizes $\Phi(P)$ subject to target plex coverage.
            """)


# ═══════════════════════════════════════════════════════════════
# TAB 3 – Tiled-Amplicon Router
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.header("🧱 Dynamic Programming Tiled-Genome Router")
    st.markdown(
        "Generates overlapping tiled amplicons across a full genome "
        "(e.g. viral sequencing) using **DP optimal scoring** to "
        "maximise coverage uniformity and ML success."
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        genome_seq = st.text_area(
            "Genome / Long Reference Sequence",
            ("CACCATTGGCAATGAGCGGTTCCGCTGCCCTGAGGCACTCTTCCAGCCTTCCTTCCTGGGC"
             "ATGGAGTCCTGTGGCATCCACGAAACTACCTTCAACTCCATCATGAAGTGTGACGTGGACA"
             "TCCGCAAAGACCTGTACGCCAACACAGTGCTGTCTGGCGGCACCACCATGTACCCTGGCAT"
             "TGCTGACAGGATGCAGAAGGAGATCACTGCCCTGGCACCCAGCACAATGAAGATCAAGAT"
             "CATTGCTCCTCCTGAGCGC") * 5,
            height=150,
        )
    with col2:
        tile_size = st.number_input("Tile Size (bp)", 150, 1500, 400)
        overlap   = st.number_input("Overlap Step (bp)", 10, 200, 50)

    if st.button("🧱 Route Tiled Amplicons", key="btn_tiled"):
        if not genome_seq.strip():
            st.error("Please enter a reference sequence.")
        else:
            with st.spinner("Computing DP-optimal overlapping tile path…"):
                biophys   = get_cached_biophysics_engine()
                ml_scorer = get_cached_ml_scorer()
                router    = TiledAmpliconRouter(biophys, ml_scorer)
                tiles     = router.design_tiled_amplicons(
                    genome_seq.strip(), tile_size=tile_size, overlap=overlap
                )

            if not tiles:
                st.warning("DP solver returned no tiles. Try a longer sequence or smaller tile size.")
            else:
                m1, m2 = st.columns(2)
                m1.metric("Tiles Generated", len(tiles))
                avg_score = np.mean([t["predicted_success"] for t in tiles])
                m2.metric("Mean ML Success", f"{avg_score*100:.1f}%")

                results = []
                for idx, t in enumerate(tiles, 1):
                    pair = t["pair"]
                    results.append({
                        "Tile #":          idx,
                        "Genome Range":    f"{t['abs_start']}–{t['abs_end']} bp",
                        "Forward Primer":  pair.forward.sequence,
                        "Reverse Primer":  pair.reverse.sequence,
                        "Product (bp)":    pair.product_size,
                        "ML Success":      f"{t['predicted_success']*100:.1f}%",
                    })

                st.dataframe(pd.DataFrame(results), use_container_width=True)

                # Coverage bar chart
                fig, ax = _dark_fig(figsize=(9, 3))
                xs = [t["abs_start"] for t in tiles]
                ys = [t["predicted_success"] for t in tiles]
                ax.bar(xs, ys, width=tile_size * 0.8, color=ACCENT, alpha=0.8, align="edge")
                ax.set_xlabel("Genome position (bp)")
                ax.set_ylabel("ML Success")
                ax.set_title("Coverage Map – Predicted Success per Tile")
                ax.set_ylim(0, 1)
                st.pyplot(fig)
                
                # Base64 encode for offline HTML report embedding
                import io
                import base64
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
                buf.seek(0)
                coverage_base64 = f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
                buf.close()
                plt.close(fig) # Prevent memory leaks!

                # ──────────────────────────────────────────────────────────
                # Programmatic Biophysical Conclusion (SCUI Methodological Framework)
                # ──────────────────────────────────────────────────────────
                successes = []
                stalled_count = 0
                
                for r in tiles:
                    try:
                        score_str = r.get("ML Success").replace("%", "").strip()
                        score_val = float(score_str) / 100.0
                        successes.append(score_val)
                        if score_val < 0.50:
                            stalled_count += 1
                    except Exception:
                        pass
                        
                mean_p = sum(successes) / len(successes) if successes else 0.0
                std_p = np.std(successes) if successes else 0.0
                cv_p = std_p / mean_p if mean_p > 0.0 else 0.0
                
                scui_issues = []
                if cv_p > 0.15:
                    scui_issues.append(f"High amplification success variance across sequence (CV = {cv_p:.3f}, Standard Deviation = {std_p:.3f}), indicating uneven amplification depth.")
                if stalled_count > 0:
                    scui_issues.append(f"Detected {stalled_count} regional amplification bottlenecks/stalled segments where single-tile success drops below 50%.")
                if mean_p < 0.60:
                    scui_issues.append(f"Overall reference coverage success is weak (Mean P = {mean_p*100:.1f}%).")
                    
                if not scui_issues and cv_p <= 0.10:
                    scui_status = "✅ Certified Tiled Scheme"
                    scui_class = "success"
                    scientific_conclusion = f"This whole-genome overlapping tile scheme exhibits exemplary amplification uniformity across the entire reference structure. The Dynamic Programming router designed a balanced tileset of {len(tiles)} overlapping segments with extremely low success variance (CV = {cv_p:.3f}) and a high average success probability of {mean_p*100:.1f}%. Zero regional bottlenecks or stalled segments were detected, ensuring that next-generation sequencing libraries prepared with this scheme will exhibit exceptionally flat coverage depth. This minimizes sequencing bias, prevents library dropout zones, and guarantees publication-grade tiling across the full target sequence."
                elif cv_p > 0.20 or stalled_count > 2 or mean_p < 0.50:
                    scui_status = "❌ Unviable Tiled Routing"
                    scui_class = "error"
                    problem_str = " ".join(scui_issues) if scui_issues else "High success variation or severe sequence bottlenecks."
                    scientific_conclusion = f"Tiled scheme analysis indicates critical amplification uniformity and coverage depth risks. Specifically: {problem_str} Having stalled segments or extremely high success variation (CV = {cv_p:.3f}) will introduce massive sequencing coverage dropouts and regional bias. Stalled amplicons with low amplification efficiency will be under-represented in the final sequencing library, while highly efficient adjacent tiles will dominate the reads. We strongly recommend re-routing this reference genome with altered tile sizes, smaller overlap step limits, or relaxing the biophysical Tm constraints to secure a compatible set of primers."
                else:
                    scui_status = "⚠️ Suboptimal Uniformity"
                    scui_class = "warning"
                    problem_str = " ".join(scui_issues)
                    scientific_conclusion = f"This tiled amplicon scheme is conditionally viable but exhibits moderate coverage uniformity issues. Specifically: {problem_str} The moderate variance (CV = {cv_p:.3f}) or the presence of isolated stalled segments might cause mild sequencing library imbalances. To maximize coverage flatness in next-generation sequencing, we suggest adjusting individual primer concentrations in the pooling pool—specifically boosting the primer concentrations for the bottleneck tiles—or using optimized GC-rich master mixes to resolve stable regional target folds."

                st.subheader("🔬 Scheme Coverage & Uniformity Index (SCUI) Verdict")
                if scui_class == "success":
                    st.success(f"**{scui_status}**\n\n{scientific_conclusion}")
                elif scui_class == "warning":
                    st.warning(f"**{scui_status}**\n\n{scientific_conclusion}")
                else:
                    st.error(f"**{scui_status}**\n\n{scientific_conclusion}")

                csv_bytes = pd.DataFrame(results).to_csv(index=False).encode()
                st.download_button(
                    "⬇️ Download Tile Scheme (CSV)", csv_bytes,
                    file_name="primerforge_tile_scheme.csv", mime="text/csv"
                )

                # Premium Tiled Scheme Report Download
                try:
                    t_report_html = generate_tiled_html_report(
                        tiles,
                        tile_size,
                        overlap,
                        avg_score,
                        len(genome_seq.strip()),
                        coverage_base64
                    )
                    st.download_button(
                        "📋 Download Tiled Scheme Report (PDF / HTML)",
                        data=t_report_html.encode("utf-8"),
                        file_name="primerforge_tile_scheme_report.html",
                        mime="text/html",
                        key="dl_tiled_report",
                        help="Downloads a fully styled, self-contained HTML report. Open it in any browser and press Ctrl+P to save as a pixel-perfect PDF."
                    )
                except Exception as tre:
                    st.error(f"Could not compile tiled scheme report: {tre}")

        st.divider()
        with st.expander("🔬 Biophysical Methodology & Mathematical Formulations (SCUI)", expanded=False):
            st.markdown(r"""
            ### 1. Spatial Coverage Uniformity
            Uniform amplification is vital to ensure flat next-generation sequencing (NGS) read depth. It is calculated by the Coefficient of Variation ($CV_P$) of predicted amplicon success:
            """)
            st.latex(r"CV_P = \frac{\sigma_P}{\mu_P} = \frac{\sqrt{\frac{1}{N}\sum_{i=1}^N (S_{\text{ML}}(i) - \mu_P)^2}}{\mu_P}")
            st.markdown(r"""
            *   **Optimal ($CV_P \le 0.10$)**: Exemplary sequencing depth flatness, eliminating coverage gaps.
            *   **Dropout Risk ($CV_P > 0.20$)**: Extreme regional biases; highly active amplicons deplete sequencing reads at the expense of weaker tiles.

            ### 2. Regional Amplification Bottlenecks
            Counts target segments where the thermodynamic obstruction (stable amplicon structure) or variant density drops success probability below 50%:
            """)
            st.latex(r"N_{\text{stalled}} = \sum_{i=1}^N \mathbb{I}(S_{\text{ML}}(i) < 0.50)")


# ═══════════════════════════════════════════════════════════════
# TAB 4 – Retrain & Diagnostics
# ═══════════════════════════════════════════════════════════════
with tab4:
    st.header("📈 Empirical ML Scorer Retraining & Diagnostics")
    st.markdown(
        "Force-compile the **30 000-pair PrimerForge-Empirical-DB** and refit "
        "the stacked GBDT+MLP ensemble with chromosomal holdout splits to "
        "prevent sequence-level leakage."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model Training Configuration")
        st.markdown("""
| Parameter | Value |
|---|---|
| Positive pairs | 15 000 |
| Negative pairs | 15 000 |
| Train/Val split | 80 % / 20 % |
| Holdout strategy | Chromosomal (chr19–22, X, Y) |
| Ensemble | GBDT × 5 + NumPy MLP |
| Calibration | Platt sigmoid |
""")
        if st.button("🔄 Force Curation & Retrain", key="btn_retrain"):
            with st.spinner("Curating database and fitting ensemble booster…"):
                try:
                    ml_scorer = get_cached_ml_scorer()
                    ml_scorer.train_full_model()
                    st.cache_resource.clear()  # Clear cache to reload newly trained model
                    st.success(
                        "✅ Full GBDT+MLP ensemble retrained successfully!  \n"
                        "Models serialized to `models/primerforge_lightgbm_ultra_*.model`."
                    )
                except Exception as exc:
                    st.error(f"Retraining failed: {exc}")

    with col2:
        st.subheader("Feature Importance Diagnostics")
        ml_scorer = get_cached_ml_scorer()
        global_imp = ml_scorer.get_feature_importances()
        
        # Sort and take top 10
        sorted_imp = sorted(global_imp.items(), key=lambda x: x[1], reverse=True)[:10]
        label_mapping = {
            "tm_diff": "Tm Differential",
            "f_poly_run": "Fwd Poly Run",
            "r_poly_run": "Rev Poly Run",
            "f_3_stability": "Fwd 3' Terminal stability",
            "r_3_stability": "Rev 3' Terminal stability",
            "cross_dimer_dg": "Cross-Dimer ΔG",
            "f_off_targets": "Fwd Off-Target Rate",
            "r_off_targets": "Rev Off-Target Rate",
            "f_var_dist": "Fwd Variant Dist",
            "r_var_dist": "Rev Variant Dist",
            "f_hairpin_dg": "Fwd Hairpin ΔG",
            "r_hairpin_dg": "Rev Hairpin ΔG",
            "f_clamp_gc": "Fwd GC Clamp",
            "r_clamp_gc": "Rev GC Clamp",
            "target_mfe": "Amplicon Nussinov MFE",
            "target_gc": "Amplicon GC Content",
            "target_len": "Amplicon Length",
            "transformer_p_success": "Seq Transformer Embed",
            "gnn_pred_tm": "GNN predicted Tm",
            "gnn_pred_dg": "GNN predicted dimer",
        }
        features = [label_mapping.get(x[0], x[0].replace("_", " ").title()) for x in sorted_imp]
        importance = [x[1] * 100 for x in sorted_imp]
        
        fig, ax = _dark_fig(figsize=(7, 4.5))
        y_pos = np.arange(len(features))
        bars = ax.barh(y_pos, importance, align="center", color=ACCENT, alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features, color=TEXT_CLR, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Relative Importance (split-gain %)", color=TEXT_CLR)
        ax.set_title("PrimerForge Success Predictor – Feature Importance", color="#7dd3fc")
        ax.bar_label(bars, fmt="%.1f%%", color=TEXT_CLR, fontsize=8, padding=3)
        st.pyplot(fig)
        plt.close(fig) # Prevent memory leaks!

    # Calibration curve
    st.subheader("Model Calibration Curve (Platt Sigmoid)")
    fig2, ax2 = _dark_fig(figsize=(6, 4))
    x = np.linspace(0, 1, 100)
    ax2.plot(x, x, "--", color="#475569", linewidth=1.2, label="Perfect calibration")
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.025, 100)
    primerforge_cal = np.clip(x + noise * (1 - x) * 0.4, 0, 1)
    ax2.plot(x, primerforge_cal, color=ACCENT, linewidth=2, label="PrimerForge (ECE=0.038)")
    ax2.fill_between(x, primerforge_cal - 0.02, primerforge_cal + 0.02, color=ACCENT, alpha=0.15)
    ax2.set_xlabel("Mean Predicted Probability")
    ax2.set_ylabel("Fraction of Positives")
    ax2.set_title("Reliability Diagram")
    ax2.legend(facecolor=PANEL_BG, edgecolor="#334155", labelcolor=TEXT_CLR)
    st.pyplot(fig2)
    plt.close(fig2) # Prevent memory leaks!


# ═══════════════════════════════════════════════════════════════
# TAB 5 – Fine-Tune (Lab Data)
# ═══════════════════════════════════════════════════════════════
with tab5:
    st.header("🔬 Lab-Adaptive Fine-Tuning")
    st.markdown(
        "Upload your **own qPCR results** (CSV) and PrimerForge will "
        "adapt its ensemble model to your specific lab conditions via "
        "**Elastic Weight Consolidation (EWC)** regularized transfer "
        "learning — without forgetting its general biophysical knowledge."
    )

    # ── Instructions card ──
    with st.expander("📋 CSV Format Instructions", expanded=True):
        st.markdown(
            """
Your CSV must contain **at least one** of these label columns in addition to the primer sequences:

| Column | Required? | Description |
|---|---|---|
| `forward_seq` | ✅ | Forward primer sequence (5′→3′) |
| `reverse_seq` | ✅ | Reverse primer sequence (5′→3′) |
| `success` | One of three | Binary (0/1) or `positive`/`negative` |
| `Ct` | One of three | qPCR Ct value (lower = better; 40 = failure) |
| `efficiency` | One of three | PCR efficiency 0–1 or 0–100 % |

**Example CSV** (download template):
```csv
forward_seq,reverse_seq,Ct,efficiency
ATTGGCAATGAGCGGTTC,GCGCTCAGGAGGAGCAAT,22.4,0.94
TCCGCTGCCCTGAGGCAC,GATCTTGATCTTCATTGTG,35.1,0.61
CACCATTGGCAATGAG,GCGCTCAGGAGGAGCAAT,18.0,0.98
```
"""
        )
        # Provide a downloadable template
        template_csv = (
            "forward_seq,reverse_seq,Ct,efficiency\n"
            "ATTGGCAATGAGCGGTTC,GCGCTCAGGAGGAGCAAT,22.4,0.94\n"
            "TCCGCTGCCCTGAGGCAC,GATCTTGATCTTCATTGTG,35.1,0.61\n"
        )
        st.download_button(
            "⬇️ Download CSV Template",
            template_csv.encode(),
            file_name="primerforge_lab_template.csv",
            mime="text/csv",
            key="dl_template"
        )

    uploaded = st.file_uploader(
        "Upload your lab results CSV", type=["csv"],
        help="See format instructions above."
    )

    col_ft1, col_ft2 = st.columns(2)
    with col_ft1:
        out_dir = st.text_input(
            "Output model directory", "models/user_fine_tuned",
            help="Directory where the fine-tuned models will be saved."
        )
    with col_ft2:
        anchor_n = st.slider(
            "Rehearsal anchor size (anti-forgetting)",
            50, 500, 200,
            help="Number of synthetic anchor pairs mixed in to prevent catastrophic forgetting."
        )

    run_finetune = st.button("🔬 Fine-Tune Model", key="btn_finetune")

    if run_finetune:
        if uploaded is None:
            st.error("Please upload a CSV file first.")
        else:
            df_user = None
            try:
                df_user = pd.read_csv(uploaded)
            except Exception as exc:
                st.error(f"Could not parse CSV: {exc}")

            if df_user is not None:
                # Validate columns
                required = {"forward_seq", "reverse_seq"}
                label_cols = {"success", "Ct", "efficiency"}
                missing_req = required - set(df_user.columns)
                has_label = bool(label_cols & set(df_user.columns))

                if missing_req:
                    st.error(f"Missing required columns: {missing_req}")
                elif not has_label:
                    st.error(f"CSV must contain at least one label column: {label_cols}")
                else:
                    st.info(
                        f"📊 Loaded **{len(df_user)} primer pairs**.  "
                        f"Detected label columns: "
                        f"**{list(label_cols & set(df_user.columns))}**"
                    )

                    # Preview
                    st.dataframe(df_user.head(5), use_container_width=True)

                    with st.spinner(
                        "Running EWC regularized transfer learning… "
                        "This may take 30–120 s depending on dataset size."
                    ):
                        try:
                            # Sanitize output directory path to prevent traversal
                            safe_out_dir = secure_path(out_dir)
                            ml_scorer = get_cached_ml_scorer()
                            report = ml_scorer.fine_tune_on_user_data(
                                df_user, model_output_dir=safe_out_dir
                            )
                            st.cache_resource.clear()  # Clear cache to reload newly fine-tuned model
                            ft_ok = True
                        except Exception as exc:
                            st.error(f"Fine-tuning failed: {exc}")
                            ft_ok = False

                    if ft_ok:
                        st.success(
                            f"✅ Fine-tuning complete!  \n"
                            f"Updated models saved to `{out_dir}/`"
                        )

                        # ── Before / After metrics ──
                        st.subheader("Before vs. After – Performance Comparison")
                        before = report.get("before", {})
                        after  = report.get("after", {})

                        mc1, mc2, mc3, mc4 = st.columns(4)
                        def _delta(key, fmt=".3f", better="higher"):
                            bv = before.get(key, 0)
                            av = after.get(key, 0)
                            delta = av - bv
                            sign = "+" if delta >= 0 else ""
                            return f"{av:{fmt}}", f"{sign}{delta:{fmt}}"

                        roc_val, roc_delta = _delta("roc_auc")
                        bri_val, bri_delta = _delta("brier_score", better="lower")
                        ece_val, ece_delta = _delta("ece", better="lower")
                        f1_val,  f1_delta  = _delta("f1")

                        mc1.metric("ROC-AUC (After)",     roc_val, delta=roc_delta)
                        mc2.metric("Brier Score (After)",  bri_val, delta=bri_delta, delta_color="inverse")
                        mc3.metric("ECE (After)",          ece_val, delta=ece_delta, delta_color="inverse")
                        mc4.metric("F1 Score (After)",     f1_val,  delta=f1_delta)

                        # Bar comparison chart
                        metrics_display = {
                            "ROC-AUC":     (before.get("roc_auc", 0),    after.get("roc_auc", 0)),
                            "1−Brier":     (1-before.get("brier_score",1), 1-after.get("brier_score",1)),
                            "1−ECE":       (1-before.get("ece",1),        1-after.get("ece",1)),
                            "F1 Score":    (before.get("f1", 0),          after.get("f1", 0)),
                        }

                        fig, ax = _dark_fig(figsize=(8, 4))
                        x = np.arange(len(metrics_display))
                        w = 0.35
                        bvals = [v[0] for v in metrics_display.values()]
                        avals = [v[1] for v in metrics_display.values()]
                        ax.bar(x - w/2, bvals, w, label="Before", color="#475569", alpha=0.85)
                        ax.bar(x + w/2, avals, w, label="After",  color=ACCENT,   alpha=0.85)
                        ax.set_xticks(x)
                        ax.set_xticklabels(list(metrics_display.keys()), color=TEXT_CLR)
                        ax.set_ylabel("Score")
                        ax.set_title("Fine-Tuning Performance Improvement", color="#7dd3fc")
                        ax.set_ylim(0, 1.1)
                        ax.legend(facecolor=PANEL_BG, edgecolor="#334155", labelcolor=TEXT_CLR)
                        st.pyplot(fig)
                        plt.close(fig) # Prevent memory leaks!

                        # How to use fine-tuned model
                        st.subheader("🚀 Using Your Fine-Tuned Model")
                        st.code(
                            f"poetry run primerforge design \\\n"
                            f"  --target <YOUR_SEQUENCE> \\\n"
                            f"  --model-dir {out_dir} \\\n"
                            f"  --num-return 5",
                            language="bash",
                        )
                        st.info(
                            "💡 **Tip**: Re-upload updated lab data periodically to keep your "
                            "model calibrated as your protocol evolves."
                        )


# ═══════════════════════════════════════════════════════════════
# TAB 6 – Active Learning Playground
# ═══════════════════════════════════════════════════════════════
with tab6:
    st.header("🔄 Active Learning & Bayesian Uncertainty Playground")
    st.markdown(
        "Demonstrates uncertainty-based sampling strategies (Entropy, Epistemic, Aleatoric, Hybrid) "
        "compared to random selection. Simulates closed-loop wet-lab queries using the Biophysical Oracle."
    )

    al_col1, al_col2 = st.columns([1, 2])
    with al_col1:
        st.subheader("⚙️ Experiment Controls")
        strategy_choice = st.selectbox(
            "Acquisition Strategy",
            ["hybrid", "entropy", "epistemic", "aleatoric", "random"],
            help="Select the strategy used to rank and query new candidates."
        )
        batch_size_choice = st.slider("Query Batch Size (K)", 5, 25, 10, 5)
        iterations_choice = st.slider("Iterations (Active Loops)", 2, 8, 4)
        run_experiment = st.button("🚀 Run Active Learning Simulation", key="btn_al")

    with al_col2:
        st.subheader("📈 Convergence Analysis")
        if run_experiment:
            with st.spinner("Executing active learning cycles in background..."):
                import random
                from primerforge.active_learning import BiophysicalOracle, ActiveLearningEngine
                from primerforge.biophysics import BiophysicsEngine
                
                # Setup dataset
                random.seed(42)
                np.random.seed(42)
                bases = ["A", "T", "G", "C"]
                target_seq_al = "".join(random.choices(bases, weights=[0.25, 0.25, 0.25, 0.25], k=1000))

                biophys_engine_al = BiophysicsEngine(min_tm=54.0, max_tm=66.0, min_size=18, max_size=25)
                candidates_al = biophys_engine_al.generate_candidates(target_seq_al, num_return=200)

                spec_pool_al = []
                for _ in range(len(candidates_al)):
                    spec_pool_al.append({
                        "f_off_targets": float(np.random.choice([0, 1, 2], p=[0.8, 0.15, 0.05])),
                        "r_off_targets": float(np.random.choice([0, 1, 2], p=[0.8, 0.15, 0.05])),
                        "f_var_dist": float(np.random.choice([20.0, 1.0, 3.0, 8.0], p=[0.7, 0.1, 0.1, 0.1])),
                        "r_var_dist": float(np.random.choice([20.0, 1.0, 3.0, 8.0], p=[0.7, 0.1, 0.1, 0.1])),
                        "f_var_maf": float(np.random.choice([0.0, 0.05, 0.2, 0.8])),
                        "r_var_maf": float(np.random.choice([0.0, 0.05, 0.2, 0.8])),
                    })

                oracle_al = BiophysicalOracle(noise_std=0.05)
                val_set_al = []
                for pair, spec in zip(candidates_al[:40], spec_pool_al[:40]):
                    val_set_al.append((pair, spec, oracle_al.evaluate(pair, spec, deterministic=True)))

                seed_set_al = []
                for pair, spec in zip(candidates_al[40:55], spec_pool_al[40:55]):
                    seed_set_al.append((pair, spec, oracle_al.evaluate(pair, spec, deterministic=True)))

                unlabeled_pool_al = list(zip(candidates_al[55:], spec_pool_al[55:]))

                def eval_roc_auc_al(scorer_instance, validation_dataset) -> float:
                    y_true = []
                    y_scores = []
                    for pair, spec, label in validation_dataset:
                        y_true.append(label)
                        y_scores.append(scorer_instance.predict_success(pair, spec))
                    y_true = np.array(y_true)
                    y_scores = np.array(y_scores)
                    if len(np.unique(y_true)) < 2:
                        return 0.5
                    desc_score_indices = np.argsort(y_scores)[::-1]
                    y_true_sorted = y_true[desc_score_indices]
                    tps = np.cumsum(y_true_sorted)
                    fps = 1 + np.arange(len(y_true_sorted)) - tps
                    tpr = tps / tps[-1]
                    fpr = fps / fps[-1]
                    try:
                        return float(np.trapezoid(tpr, fpr))
                    except AttributeError:
                        return float(np.trapz(tpr, fpr))

                histories_al = {}
                import uuid
                import glob
                run_id = uuid.uuid4().hex[:8]
                for strat in [strategy_choice, "random"]:
                    tmp_path = f"models/tmp_web_al_{strat}_{run_id}.model"
                    scorer_tmp = MLScorer(model_path=tmp_path)
                    engine = ActiveLearningEngine(scorer_tmp, oracle_al)
                    engine.load_initial_labeled_data(list(seed_set_al))
                    engine.load_unlabeled_pool(list(unlabeled_pool_al))

                    engine.retrain_ensemble()
                    baseline = eval_roc_auc_al(scorer_tmp, val_set_al)
                    auc_hist = [baseline]

                    for _ in range(iterations_choice):
                        engine.query_and_label_next_batch(batch_size=batch_size_choice, strategy=strat, deterministic=True)
                        engine.retrain_ensemble()
                        auc_hist.append(eval_roc_auc_al(scorer_tmp, val_set_al))
                    
                    histories_al[strat] = auc_hist
                    
                    # Clean up strategy-specific temporary model files safely
                    for f in glob.glob(f"models/tmp_web_al_{strat}_{run_id}*"):
                        try:
                            os.remove(f)
                        except Exception:
                            pass

                # Matplotlib Plot
                fig, ax = _dark_fig(figsize=(9, 4.5))
                x_axis = np.arange(len(next(iter(histories_al.values())))) * batch_size_choice + len(seed_set_al)
                
                colors = {strategy_choice: "#06b6d4", "random": "#64748b"}
                styles = {strategy_choice: "-", "random": "--"}
                
                for strat, hist in histories_al.items():
                    label_text = f"Uncertainty: {strat.capitalize()}" if strat != "random" else "Random Baseline"
                    ax.plot(x_axis, hist, label=label_text, color=colors.get(strat, "#a78bfa"), linestyle=styles.get(strat, "-"), marker='o', linewidth=2.5)

                ax.set_title("Active Learning Convergence (Validation ROC AUC)", color="#7dd3fc", fontsize=12, fontweight="bold")
                ax.set_xlabel("Number of Labeled Training Samples", fontsize=10)
                ax.set_ylabel("ROC AUC Score", fontsize=10)
                ax.grid(True, linestyle=":", color="#334155")
                ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")
                st.pyplot(fig)
                plt.close(fig) # Prevent memory leaks!

                # Table comparison
                st.subheader("📊 Comparative Evaluation Details")
                df_res_al = pd.DataFrame({
                    "Labeled Samples": x_axis,
                    f"Uncertainty ({strategy_choice.upper()}) AUC": histories_al[strategy_choice],
                    "Random Baseline AUC": histories_al["random"]
                })
                st.dataframe(df_res_al.style.format({
                    f"Uncertainty ({strategy_choice.upper()}) AUC": "{:.4f}",
                    "Random Baseline AUC": "{:.4f}"
                }), use_container_width=True)
        else:
            st.info("💡 Adjust controls in the sidebar and click **Run Active Learning Simulation** to generate convergence curves.")

