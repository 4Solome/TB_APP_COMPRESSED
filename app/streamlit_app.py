import textwrap

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import torch

from utils import (
    load_feature_names,
    load_feature_to_modality,
    load_model_config,
    load_preprocessor,
    load_ttvae,
    load_cluster_model,
    load_pseudotime_bounds,
    load_ood_threshold,
    get_feature_groups,
    get_all_raw_columns,
    prepare_input_dataframe,
    transform_input,
    compute_latent,
    compute_pseudotime,
    assign_cluster,
    batched_reconstruction_error,
    decode_synthetic_from_transformed,
)

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="TB RiskLens Lite",
    page_icon="🫁",
    layout="wide",
)

# ============================================================
# CUSTOM UI STYLING
# ============================================================
st.markdown(
    """
    <style>
    :root {
        --bg: #060b17;
        --card: rgba(13, 20, 38, 0.88);
        --border: rgba(114, 137, 218, 0.22);
        --text: #eef2ff;
        --muted: #a8b2d1;
        --pink: #ff4d8d;
        --purple: #7c4dff;
        --blue: #3b82f6;
        --green: #10b981;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(76, 29, 149, 0.20), transparent 25%),
            radial-gradient(circle at top right, rgba(37, 99, 235, 0.14), transparent 25%),
            linear-gradient(180deg, #040915 0%, #050b17 50%, #06101d 100%);
        color: var(--text);
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1320px;
    }

    h1, h2, h3, h4, h5, h6, p, label, span {
        color: var(--text);
    }

    .section-card {
        background: linear-gradient(180deg, rgba(13,20,38,0.96), rgba(8,14,28,0.98));
        border: 1px solid rgba(114, 137, 218, 0.20);
        border-radius: 24px;
        padding: 1.15rem 1.15rem 1.25rem 1.15rem;
        box-shadow: 0 18px 40px rgba(0, 0, 0, 0.22);
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    .section-title {
        font-size: 1.7rem;
        font-weight: 800;
        margin: 0;
    }

    .section-subtitle {
        color: var(--muted);
        margin-top: 0.2rem;
        font-size: 1rem;
    }

    .metric-card {
        background: linear-gradient(180deg, rgba(14, 22, 41, 0.96), rgba(10, 15, 28, 0.96));
        border: 1px solid rgba(114, 137, 218, 0.18);
        border-radius: 18px;
        padding: 0.35rem;
        box-shadow: 0 10px 24px rgba(0,0,0,0.16);
    }

    .mapping-card-label {
        background: rgba(10, 16, 30, 0.72);
        border: 1px solid rgba(114, 137, 218, 0.20);
        border-radius: 16px;
        padding: 0.7rem 0.75rem;
        margin-bottom: 0.35rem;
        font-weight: 800;
        font-size: 0.94rem;
        color: #eef2ff;
    }

    .stButton > button {
        border: none !important;
        border-radius: 14px !important;
        padding: 0.78rem 1.35rem !important;
        font-weight: 700 !important;
        color: white !important;
        background: linear-gradient(90deg, var(--pink), var(--purple)) !important;
        box-shadow: 0 10px 25px rgba(124,77,255,0.26) !important;
    }

    div[data-testid="stDownloadButton"] > button {
        background: linear-gradient(90deg, var(--green), #059669) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
    }

    div[data-testid="stFileUploader"] {
        background: rgba(10, 16, 30, 0.72);
        border: 1px solid rgba(114, 137, 218, 0.18);
        border-radius: 18px;
        padding: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# CLUSTER DEFINITIONS FOR LIGHTWEIGHT MODEL
# Ordered from highest to lowest pseudotime risk:
# 0 -> 2 -> 4 -> 1 -> 3
# ============================================================
CLUSTER_INFO = {
    0: {
        "name": "High-Risk Symptomatic Diagnostic-Confirmed Phenotype",
        "short_name": "High-Risk Confirmed/Symptomatic",
        "risk": "High Risk",
        "summary": (
            "Highest latent risk position, dominated by strong symptomatic and diagnostic TB signal. "
            "This group should be interpreted as the strongest TB-risk phenotype."
        ),
        "key_features": ["cough", "sputum", "chest_pain", "xrayres", "zn"],
    },
    2: {
        "name": "Intermediate-Risk Diagnostic-Missingness Phenotype",
        "short_name": "Intermediate-Risk Missing-Diagnostics",
        "risk": "Elevated Risk",
        "summary": (
            "Elevated pseudotime profile characterized by structured diagnostic missingness and moderate clinical signal. "
            "This may reflect patients who sit within a diagnostic escalation or incomplete investigation pathway."
        ),
        "key_features": ["zn_missing", "hiv_missing", "xray_missing", "genexpert_missing"],
    },
    4: {
        "name": "Moderate-Risk Symptomatic Screening Phenotype",
        "short_name": "Moderate-Risk Symptomatic",
        "risk": "Moderate Risk",
        "summary": (
            "Near-neutral latent risk profile with some symptom burden but weaker diagnostic confirmation signal."
        ),
        "key_features": ["cough", "chest_pain", "fever", "weight_loss"],
    },
    1: {
        "name": "Low-Risk Normal-Radiology Screening Phenotype",
        "short_name": "Low-Risk Normal Screening",
        "risk": "Low Risk",
        "summary": (
            "Low pseudotime profile associated with normal screening patterns and limited evidence of diagnostic escalation."
        ),
        "key_features": ["normal_xray", "minimal_symptoms"],
    },
    3: {
        "name": "Lowest-Risk Asymptomatic Non-Smoking Phenotype",
        "short_name": "Lowest-Risk Asymptomatic",
        "risk": "Very Low Risk",
        "summary": (
            "Lowest latent risk position, typically reflecting minimal symptom burden and stable screening profile."
        ),
        "key_features": ["minimal_symptoms", "non_smoking", "stable_profile"],
    },
}

CLUSTER_ORDER = [0, 2, 4, 1, 3]

MAX_ROWS = 30000
BATCH_SIZE = 1024

# ============================================================
# CACHED LOADERS
# ============================================================
@st.cache_resource
def load_all_artifacts():
    feature_names = load_feature_names()
    preprocessor = load_preprocessor()
    config = load_model_config()
    continuous_cols, binary_cols, categorical_cols = get_feature_groups(preprocessor, config)
    all_cols = continuous_cols + binary_cols + categorical_cols

    model = load_ttvae(input_dim=len(feature_names))
    kmeans = load_cluster_model()
    pt_bounds = load_pseudotime_bounds()
    ood_info = load_ood_threshold()
    feature_to_modality = load_feature_to_modality()

    return {
        "feature_names": feature_names,
        "preprocessor": preprocessor,
        "config": config,
        "continuous_cols": continuous_cols,
        "binary_cols": binary_cols,
        "categorical_cols": categorical_cols,
        "all_cols": all_cols,
        "model": model,
        "kmeans": kmeans,
        "pt_bounds": pt_bounds,
        "ood_info": ood_info,
        "feature_to_modality": feature_to_modality,
    }


try:
    ART = load_all_artifacts()
except Exception as e:
    st.error(
        "Model artifacts could not be loaded. Confirm that the models folder contains the lightweight files: "
        "ttvae_lite_fp16.pth, preprocessor_lite.joblib, kmeans_lite_model.joblib, "
        "feature_names_lite.json, model_config_lite.json, ood_threshold_lite.json, "
        "and pseudotime_bounds_lite.json."
    )
    st.exception(e)
    st.stop()

feature_names = ART["feature_names"]
preprocessor = ART["preprocessor"]
model = ART["model"]
kmeans = ART["kmeans"]
PT_BOUNDS = ART["pt_bounds"]
OOD_INFO = ART["ood_info"]

CONTINUOUS_COLS = ART["continuous_cols"]
BINARY_COLS = ART["binary_cols"]
CATEGORICAL_COLS = ART["categorical_cols"]
ALL_COLS = ART["all_cols"]

OOD_THRESHOLD = float(OOD_INFO.get("threshold", OOD_INFO.get("ood_threshold", 0.0)))
OOD_PERCENTILE = int(OOD_INFO.get("percentile", 95))

LATENT_DIM = int(ART["config"].get("latent_dim", 16))

# Show all selected lightweight variables for mapping.
DISPLAY_COLS = ALL_COLS

# ============================================================
# HELPERS
# ============================================================
def normalize_column_name(col_name: str) -> str:
    return "".join(ch.lower() for ch in str(col_name).strip() if ch.isalnum())


def guess_column_mapping(uploaded_columns, expected_columns):
    normalised_uploaded = {normalize_column_name(col): col for col in uploaded_columns}
    mapping = {}
    for expected in expected_columns:
        key = normalize_column_name(expected)
        mapping[expected] = normalised_uploaded.get(key, "-- Not available --")
    return mapping


def apply_hospital_column_mapping(df_raw: pd.DataFrame, mapping: dict):
    df_mapped = pd.DataFrame(index=df_raw.index)
    mapped_expected_cols = []

    for expected_col in ALL_COLS:
        source_col = mapping.get(expected_col, "-- Not available --")
        if source_col != "-- Not available --" and source_col in df_raw.columns:
            df_mapped[expected_col] = df_raw[source_col]
            mapped_expected_cols.append(expected_col)
        else:
            df_mapped[expected_col] = np.nan

    missing_expected_cols = [col for col in ALL_COLS if col not in mapped_expected_cols]
    return df_mapped, mapped_expected_cols, missing_expected_cols


def show_mapping_quality(mapped_expected_cols, missing_expected_cols):
    c1, c2, c3 = st.columns(3)
    c1.metric("Model Variables", len(ALL_COLS))
    c2.metric("Mapped Variables", len(mapped_expected_cols))
    c3.metric("Unmapped Variables", len(missing_expected_cols))

    if len(mapped_expected_cols) == 0:
        st.error("No model variables have been mapped. Please map at least one column before analysis.")
    elif missing_expected_cols:
        st.warning("Some model variables were not mapped. They will be treated as missing by the saved preprocessor.")
        with st.expander("View unmapped model variables", expanded=False):
            st.write(missing_expected_cols)
    else:
        st.success("All lightweight model variables have been mapped successfully.")


def progression_position_label(pt_norm: float) -> str:
    # Higher pseudotime corresponds to stronger clinical/bacteriological TB signal.
    if pt_norm >= 0.80:
        return "High Latent TB Risk Position"
    if pt_norm >= 0.60:
        return "Elevated Latent TB Risk Position"
    if pt_norm >= 0.40:
        return "Moderate Latent TB Risk Position"
    if pt_norm >= 0.20:
        return "Low Latent TB Risk Position"
    return "Very Low Latent TB Risk Position"


def risk_bucket_from_cluster(cluster_id: int) -> str:
    return CLUSTER_INFO.get(cluster_id, {}).get("risk", "Unknown")


def build_patient_results(latents, pseudotime_norm, clusters, rec_error, ood_flags):
    rows = []

    for i in range(len(clusters)):
        if bool(ood_flags[i]):
            rows.append(
                {
                    "Cluster": "—",
                    "Clinical Profile": "—",
                    "Risk Group": "—",
                    "Pseudotime Score": round(float(pseudotime_norm[i]), 4),
                    "Risk Position": "—",
                    "Reconstruction Error": round(float(rec_error[i]), 6),
                    "Reliability": "⚠️ OOD - not assessed",
                }
            )
            continue

        cid = int(clusters[i])
        info = CLUSTER_INFO.get(cid, {})

        rows.append(
            {
                "Cluster": cid,
                "Clinical Profile": info.get("short_name", info.get("name", "Clinical profile")),
                "Risk Group": risk_bucket_from_cluster(cid),
                "Pseudotime Score": round(float(pseudotime_norm[i]), 4),
                "Risk Position": progression_position_label(float(pseudotime_norm[i])),
                "Reconstruction Error": round(float(rec_error[i]), 6),
                "Reliability": "✅ Within training distribution",
            }
        )

    return pd.DataFrame(rows)


def plot_profile_distribution(results_df):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    fig.patch.set_facecolor("#0b1324")
    ax.set_facecolor("#0b1324")

    assessed = results_df[results_df["Reliability"].str.startswith("✅", na=False)]
    if assessed.empty:
        ax.text(0.5, 0.5, "No assessed records", ha="center", va="center", color="white")
        ax.set_axis_off()
        return fig

    ordered = assessed["Clinical Profile"].value_counts()
    ordered.plot(kind="bar", ax=ax)
    ax.set_ylabel("Number of Patients", color="white")
    ax.set_title("Clinical Profile Distribution", color="white")
    ax.tick_params(axis="x", colors="white", labelrotation=35)
    ax.tick_params(axis="y", colors="white")
    for label in ax.get_xticklabels():
        label.set_ha("right")
    for spine in ax.spines.values():
        spine.set_color("#2a3550")
    fig.tight_layout(pad=1.8)
    return fig


def build_profile_summary(results_df):
    assessed = results_df[results_df["Reliability"].str.startswith("✅", na=False)]
    if assessed.empty:
        return pd.DataFrame(columns=["Clinical Profile", "Risk Group", "Patients"])

    return (
        assessed.groupby(["Clinical Profile", "Risk Group"], as_index=False)
        .agg(Patients=("Clinical Profile", "count"))
        .sort_values("Patients", ascending=False)
    )


# ============================================================
# HERO SECTION
# ============================================================
left_col, right_col = st.columns([1.6, 1], gap="large")

with left_col:
    st.markdown("### ✦ AI-POWERED")
    st.markdown("# TB RiskLens Lite")
    st.markdown(
        """
        Lightweight compressed TTVAE deployment for cohort-level tuberculosis
        latent risk profiling, phenotype discovery, and pseudotime-inspired sequencing.
        """
    )

    st.markdown("### Key Capabilities")
    st.markdown("**🧠 Latent Phenotyping**  \nDiscover TB-related patient profiles from structured survey data.")
    st.markdown("**📈 Pseudotime Risk Sequencing**  \nEstimate relative latent risk position across patient cohorts.")
    st.markdown("**🛡 Reliability Auditing**  \nFlag out-of-distribution records using reconstruction error.")

with right_col:
    st.markdown("<div style='height: 70px;'></div>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center; font-size: 120px;'>🫁</div>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; margin-top: -10px;'>Compressed TTVAE</h3>", unsafe_allow_html=True)

# ============================================================
# MODEL INFO
# ============================================================
with st.expander("Model information", expanded=False):
    st.write(
        {
            "Encoded features": len(feature_names),
            "Raw model variables": len(ALL_COLS),
            "Continuous variables": len(CONTINUOUS_COLS),
            "Binary variables": len(BINARY_COLS),
            "Categorical variables": len(CATEGORICAL_COLS),
            "Latent dimension": LATENT_DIM,
            "OOD percentile": OOD_PERCENTILE,
            "OOD threshold": OOD_THRESHOLD,
        }
    )
    st.write("Model variables:")
    st.write(ALL_COLS)

# ============================================================
# UPLOAD + ANALYSIS
# ============================================================
st.markdown(
    """
    <div class="section-card">
        <div class="section-title">Upload Patient Cohort</div>
        <div class="section-subtitle">
            Upload a CSV file, map its columns to the lightweight model variables,
            and run TB latent risk profiling.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload a CSV file containing TB patient records",
    type=["csv"],
)

results = None
df_clean = None
df_raw = None
mapping = {}
analyze = False

if uploaded_file is not None:
    try:
        df_raw = pd.read_csv(uploaded_file)

        if len(df_raw) == 0:
            st.error("The uploaded CSV file has no patient records.")
            st.stop()

        if len(df_raw) > MAX_ROWS:
            st.error(f"The uploaded dataset contains {len(df_raw):,} rows. This app supports up to {MAX_ROWS:,} rows per run.")
            st.stop()

        st.success(f"CSV uploaded successfully: {len(df_raw):,} records and {len(df_raw.columns):,} columns detected.")

        with st.expander("Preview uploaded CSV", expanded=False):
            st.dataframe(df_raw.head(20), use_container_width=True)

        st.markdown("### Column Setup")

        auto_mapping = guess_column_mapping(df_raw.columns, ALL_COLS)
        available_options = ["-- Not available --"] + list(df_raw.columns)

        schema_mode = st.radio(
            "Column setup",
            options=["Use model names", "Map hospital variables"],
            horizontal=True,
        )

        if schema_mode == "Use model names":
            mapping = auto_mapping
        else:
            st.caption("Map each lightweight model variable to the matching column in the uploaded CSV.")
            mapping = {}
            grid_cols_per_row = 3

            for start_idx in range(0, len(ALL_COLS), grid_cols_per_row):
                row_variables = ALL_COLS[start_idx:start_idx + grid_cols_per_row]
                cols = st.columns(grid_cols_per_row, gap="medium")

                for i, expected_col in enumerate(row_variables):
                    with cols[i]:
                        default_source = auto_mapping.get(expected_col, "-- Not available --")
                        default_index = available_options.index(default_source) if default_source in available_options else 0

                        st.markdown(f"""<div class="mapping-card-label">{expected_col}</div>""", unsafe_allow_html=True)

                        mapping[expected_col] = st.selectbox(
                            label=f"Select column for {expected_col}",
                            options=available_options,
                            index=default_index,
                            key=f"map_{expected_col}",
                            label_visibility="collapsed",
                        )

        df_mapped_preview, mapped_expected_cols, missing_expected_cols = apply_hospital_column_mapping(df_raw, mapping)
        show_mapping_quality(mapped_expected_cols, missing_expected_cols)

        with st.expander("Preview mapped input", expanded=False):
            st.dataframe(df_mapped_preview[ALL_COLS].head(20), use_container_width=True)

        analyze = st.button(
            "Run Analysis",
            type="primary",
            disabled=(len(mapped_expected_cols) == 0),
        )

    except Exception as e:
        st.error("The uploaded CSV file could not be read. Please confirm that it is a valid CSV file.")
        st.exception(e)
        st.stop()

if uploaded_file is not None and analyze:
    try:
        df_mapped, mapped_expected_cols, missing_expected_cols = apply_hospital_column_mapping(df_raw, mapping)

        df_clean, X = transform_input(df_mapped, preprocessor, feature_names)
        latents = compute_latent(model, X)
        pseudotime_norm = compute_pseudotime(latents, bounds=PT_BOUNDS)
        clusters = assign_cluster(kmeans, latents)

        rec_error = batched_reconstruction_error(model, X, batch_size=BATCH_SIZE)
        ood_flags = rec_error > OOD_THRESHOLD

        results = build_patient_results(
            latents=latents,
            pseudotime_norm=pseudotime_norm,
            clusters=clusters,
            rec_error=rec_error,
            ood_flags=ood_flags,
        )

        st.success("Cohort processed successfully.")

    except Exception as e:
        st.error(
            "The uploaded data could not be processed. Confirm that mapped columns are compatible "
            "with the lightweight model schema and saved preprocessing pipeline."
        )
        st.exception(e)
        st.stop()

# ============================================================
# RESULTS
# ============================================================
if results is not None:
    st.markdown("### Cohort Summary")

    assessed_count = int(results["Reliability"].str.startswith("✅", na=False).sum())
    ood_count = int(results["Reliability"].str.startswith("⚠️", na=False).sum())

    m1, m2, m3, m4 = st.columns(4, gap="medium")
    with m1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Records", f"{len(results):,}")
        st.markdown("</div>", unsafe_allow_html=True)
    with m2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Assessed Records", f"{assessed_count:,}")
        st.markdown("</div>", unsafe_allow_html=True)
    with m3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Not Assessed / OOD", f"{ood_count:,}")
        st.markdown("</div>", unsafe_allow_html=True)
    with m4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        assessed_profiles = results.loc[results["Reliability"].str.startswith("✅", na=False), "Clinical Profile"].nunique()
        st.metric("Profiles Detected", int(assessed_profiles))
        st.markdown("</div>", unsafe_allow_html=True)

    if ood_count > 0:
        st.warning(
            "Some records were outside the training distribution. For safety, the system did not assign "
            "a clinical profile, risk group, or final risk position to those records."
        )

    st.markdown("### Patient-Level Results")
    st.dataframe(results, use_container_width=True)

    st.markdown("### Distribution View")
    st.pyplot(plot_profile_distribution(results), use_container_width=True)

    with st.expander("Risk Profile Summary", expanded=True):
        st.dataframe(build_profile_summary(results), use_container_width=True)

    with st.expander("Clinical Profile Definitions", expanded=False):
        for cid in CLUSTER_ORDER:
            info = CLUSTER_INFO[cid]
            st.markdown(
                f"**Cluster {cid}: {info['name']}**  \n"
                f"- Risk group: {info['risk']}  \n"
                f"- Interpretation: {info['summary']}  \n"
                f"- Key signals: {', '.join(info['key_features'])}"
            )

    with st.expander("Mapped Data Preview", expanded=False):
        preview_cols = [col for col in ALL_COLS if col in df_clean.columns]
        st.dataframe(df_clean[preview_cols].head(200), use_container_width=True)

    st.download_button(
        "Download Patient-Level Results",
        results.to_csv(index=False),
        file_name="tb_risk_results_lite.csv",
        mime="text/csv",
    )

# ============================================================
# SYNTHETIC DATA GENERATION
# ============================================================
st.markdown(
    textwrap.dedent("""
    <div class="section-card">
        <div class="section-title">Synthetic Patient Generation</div>
        <div class="section-subtitle">
            Generate synthetic profiles from the lightweight TTVAE decoder for testing and demonstration.
        </div>
    </div>
    """),
    unsafe_allow_html=True,
)

col1, col2 = st.columns([1.2, 1])
with col1:
    num_samples = st.slider("Number of synthetic patients", 10, 200, 50)
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    generate_clicked = st.button("Generate Synthetic Patients")

if generate_clicked:
    z = torch.randn(num_samples, LATENT_DIM)

    with torch.no_grad():
        syn_array = model.decode(z).cpu().numpy()

    syn_df = pd.DataFrame(syn_array, columns=feature_names)
    decoded = decode_synthetic_from_transformed(
        syn_df,
        continuous_cols=CONTINUOUS_COLS,
        binary_cols=BINARY_COLS,
        categorical_cols=CATEGORICAL_COLS,
    )

    st.success(f"Generated {num_samples} synthetic patient records.")
    st.markdown("### Preview of Generated Data")
    st.dataframe(decoded.head(10), use_container_width=True)

    st.download_button(
        "Download Synthetic Dataset",
        decoded.to_csv(index=False),
        file_name="synthetic_tb_patients_lite.csv",
        mime="text/csv",
    )

st.markdown(
    """
    <div style="
        margin-top: 1rem;
        padding: 0.9rem;
        border-radius: 16px;
        background: rgba(8, 13, 26, 0.6);
        border: 1px solid rgba(114, 137, 218, 0.18);
        color: #a8b2d1;
        font-size: 14px;
    ">
        ⚠️ This system is for cohort-level analytical support only. It is not a diagnostic tool.
        Synthetic records are statistically plausible generated examples, not real patients.
    </div>
    """,
    unsafe_allow_html=True,
)
