import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

# ============================================================
# IMPORT TTVAE MODEL FROM SAME APP FOLDER
# ============================================================
APP_DIR = Path(__file__).resolve().parent

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from ttvae_model import TTVAE

# ============================================================
# PATHS
# ============================================================
DEVICE = torch.device("cpu")

ROOT_DIR = APP_DIR.parent
MODELS_DIR = ROOT_DIR / "models"

MODEL_PATH = MODELS_DIR / "ttvae_lite_fp16.pth"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor_lite.joblib"
KMEANS_PATH = MODELS_DIR / "kmeans_lite_model.joblib"

FEATURE_NAMES_PATH = MODELS_DIR / "feature_names_lite.json"
FEATURE_MODALITY_PATH = MODELS_DIR / "feature_to_modality_lite.json"
MODEL_CONFIG_PATH = MODELS_DIR / "model_config_lite.json"
OOD_PATH = MODELS_DIR / "ood_threshold_lite.json"
PSEUDOTIME_PATH = MODELS_DIR / "pseudotime_bounds_lite.json"

MISSING_MARKERS = [
    "",
    " ",
    "na",
    "nan",
    "none",
    "missing",
    "MISSING",
    "NA",
    "NaN",
    "None",
]

# ============================================================
# JSON HELPERS
# ============================================================
def load_json(path, default=None):
    if not path.exists():
        return {} if default is None else default

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ============================================================
# ARTIFACT LOADERS
# ============================================================
def load_feature_names():
    return load_json(FEATURE_NAMES_PATH, default=[])


def load_feature_to_modality():
    return load_json(FEATURE_MODALITY_PATH, default={})


def load_model_config():
    return load_json(MODEL_CONFIG_PATH, default={})


def load_preprocessor():
    return joblib.load(PREPROCESSOR_PATH)


def load_cluster_model():
    return joblib.load(KMEANS_PATH)


def load_pseudotime_bounds():
    return load_json(PSEUDOTIME_PATH, default={})


def load_ood_threshold():
    return load_json(OOD_PATH, default={})

# ============================================================
# FEATURE GROUP EXTRACTION
# ============================================================
def _extract_columns(preprocessor, transformer_name):
    transformers = getattr(preprocessor, "transformers_", None)

    if transformers is None:
        transformers = getattr(preprocessor, "transformers", [])

    for name, transformer, cols in transformers:
        if name == transformer_name:
            return list(cols)

    return []


def get_feature_groups(preprocessor=None, config=None):
    """
    Returns only the lightweight model raw variables.
    Expected final structure:
    6 continuous, 13 binary, and 8 categorical groups.
    """

    config = config or load_model_config()
    preprocessor = preprocessor or load_preprocessor()

    continuous_cols = (
        config.get("continuous_cols")
        or config.get("cont_cols")
        or config.get("continuous_features")
        or _extract_columns(preprocessor, "cont")
    )

    binary_cols = (
        config.get("binary_cols")
        or config.get("bin_cols")
        or config.get("binary_features")
        or _extract_columns(preprocessor, "bin")
    )

    categorical_cols = (
        config.get("categorical_cols")
        or config.get("cat_cols")
        or config.get("categorical_features")
        or _extract_columns(preprocessor, "cat")
    )

    return list(continuous_cols), list(binary_cols), list(categorical_cols)


def get_all_raw_columns(preprocessor=None, config=None):
    """
    Returns all raw variables required by the lightweight model.
    """

    continuous_cols, binary_cols, categorical_cols = get_feature_groups(
        preprocessor=preprocessor,
        config=config,
    )

    return continuous_cols + binary_cols + categorical_cols

# ============================================================
# DECODER STRUCTURE
# ============================================================
def get_decoder_structure(feature_names=None, config=None):
    """
    Lightweight final model decoder structure:
    6 continuous + 13 binary + categorical groups.
    """

    config = config or load_model_config()

    n_cont = int(
        config.get(
            "n_cont",
            config.get("num_continuous", 6),
        )
    )

    n_bin = int(
        config.get(
            "n_bin",
            config.get("num_binary", 13),
        )
    )

    cat_sizes = config.get(
        "cat_sizes",
        config.get("categorical_sizes", [4, 10, 4, 4, 3, 4, 3, 4]),
    )

    cat_sizes = [int(x) for x in cat_sizes]

    return n_cont, n_bin, cat_sizes

# ============================================================
# INPUT CLEANING
# ============================================================
def standardize_missing(df):
    return df.replace(MISSING_MARKERS, np.nan)


def coerce_types(df, continuous_cols, binary_cols, categorical_cols):
    df = df.copy()

    for col in continuous_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].replace(MISSING_MARKERS, np.nan),
                errors="coerce",
            )

    for col in binary_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].replace(MISSING_MARKERS, np.nan),
                errors="coerce",
            ).clip(0, 1)

    for col in categorical_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .replace(MISSING_MARKERS, np.nan)
                .astype("object")
            )

    return df


def prepare_input_dataframe(df_raw, preprocessor=None):
    """
    Prepares uploaded raw data using the exact lightweight model input schema.
    Missing variables are created as NaN and handled by the saved preprocessor.
    """

    preprocessor = preprocessor or load_preprocessor()
    config = load_model_config()

    continuous_cols, binary_cols, categorical_cols = get_feature_groups(
        preprocessor=preprocessor,
        config=config,
    )

    all_cols = continuous_cols + binary_cols + categorical_cols

    df = df_raw.copy()
    df = standardize_missing(df)

    for col in ["household", "hhid", "household_id"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    for col in all_cols:
        if col not in df.columns:
            df[col] = np.nan

    df = df[all_cols]

    df = coerce_types(
        df,
        continuous_cols,
        binary_cols,
        categorical_cols,
    )

    return df

# ============================================================
# LOAD LIGHTWEIGHT TTVAE
# ============================================================
def load_ttvae(input_dim=None):
    feature_names = load_feature_names()
    config = load_model_config()

    if input_dim is None:
        input_dim = int(config.get("input_dim", len(feature_names)))

    n_cont, n_bin, cat_sizes = get_decoder_structure(
        feature_names=feature_names,
        config=config,
    )

    model = TTVAE(
        input_dim=input_dim,
        latent_dim=int(config.get("latent_dim", 16)),
        d_model=int(config.get("d_model", 64)),
        nhead=int(config.get("nhead", config.get("num_heads", 4))),
        n_layers=int(config.get("n_layers", config.get("num_layers", 2))),
        n_cont=n_cont,
        n_bin=n_bin,
        cat_sizes=cat_sizes,
        dropout=float(config.get("dropout", 0.1)),
    ).to(DEVICE)

    state = torch.load(MODEL_PATH, map_location=DEVICE)

    if isinstance(state, dict):
        if "model_state_dict" in state:
            state = state["model_state_dict"]

        if "state_dict" in state:
            state = state["state_dict"]

    model.load_state_dict(state, strict=True)
    model.eval()

    return model

# ============================================================
# TRANSFORM INPUT
# ============================================================
def transform_input(df_raw, preprocessor, feature_names):
    df = prepare_input_dataframe(
        df_raw,
        preprocessor=preprocessor,
    )

    X = preprocessor.transform(df)

    X_df = pd.DataFrame(
        X,
        columns=preprocessor.get_feature_names_out(),
    )

    X_df = X_df.reindex(
        columns=feature_names,
        fill_value=0.0,
    )

    return df, X_df.values.astype(np.float32)

# ============================================================
# LATENT REPRESENTATION
# ============================================================
def compute_latent(model, X):
    X_t = torch.tensor(
        X,
        dtype=torch.float32,
        device=DEVICE,
    )

    with torch.no_grad():
        mu, _ = model.encode(X_t)

    return mu.cpu().numpy()

# ============================================================
# PSEUDOTIME
# ============================================================
def compute_pseudotime(latents, bounds=None):
    bounds = bounds or {}

    if (
        "pca_mean" in bounds
        and (
            "pca_component" in bounds
            or "pca_components" in bounds
        )
    ):
        mean = np.asarray(bounds["pca_mean"], dtype=float)

        component = np.asarray(
            bounds.get(
                "pca_component",
                bounds.get("pca_components"),
            ),
            dtype=float,
        )

        if component.ndim > 1:
            component = component[0]

        pt_raw = (latents - mean) @ component

    else:
        if len(latents) < 2:
            pt_raw = latents[:, 0]
        else:
            pt_raw = PCA(
                n_components=1,
                random_state=42,
            ).fit_transform(latents).ravel()

    pmin = float(bounds.get("min", np.nanmin(pt_raw)))
    pmax = float(bounds.get("max", np.nanmax(pt_raw)))

    pt_norm = (pt_raw - pmin) / (pmax - pmin + 1e-10)

    return np.clip(pt_norm, 0.0, 1.0)

# ============================================================
# CLUSTER ASSIGNMENT
# ============================================================
def assign_cluster(kmeans, latents):
    latents = np.asarray(
        latents,
        dtype=kmeans.cluster_centers_.dtype,
    )

    return kmeans.predict(latents)

# ============================================================
# RECONSTRUCTION ERROR / OOD
# ============================================================
def batched_reconstruction_error(model, X, batch_size=1024):
    errors = []

    model.eval()

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(
                X[i:i + batch_size],
                dtype=torch.float32,
                device=DEVICE,
            )

            mu, logvar = model.encode(xb)
            z = model.reparameterize(mu, logvar)
            rec = model.decode(z)

            err = ((rec.cpu().numpy() - X[i:i + batch_size]) ** 2).mean(axis=1)

            errors.append(err)

    return np.concatenate(errors)

# ============================================================
# SYNTHETIC DECODING
# ============================================================
def decode_synthetic_from_transformed(
    syn_df,
    continuous_cols,
    binary_cols,
    categorical_cols,
):
    decoded = pd.DataFrame(index=syn_df.index)

    for col in continuous_cols:
        tcol = f"cont__{col}"

        if tcol in syn_df.columns:
            decoded[col] = (
                syn_df[tcol]
                .clip(0, 1)
                .round(3)
            )

    for col in binary_cols:
        tcol = f"bin__{col}"

        if tcol in syn_df.columns:
            decoded[col] = (
                syn_df[tcol] >= 0.5
            ).astype(int)

    for col in categorical_cols:
        prefix = f"cat__{col}_"

        matches = [
            c for c in syn_df.columns
            if str(c).startswith(prefix)
        ]

        if matches:
            decoded[col] = (
                syn_df[matches]
                .idxmax(axis=1)
                .str.replace(prefix, "", regex=False)
                .str.replace(".0", "", regex=False)
            )

    return decoded
