import json
import joblib
import numpy as np
import torch
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from ttvae_model import TTVAE

device = torch.device("cpu")

# ============================================================
# TRAINING-CONSISTENT METADATA
# ============================================================
MISSING_MARKERS = ["", " ", "na", "nan", "none", "missing", "MISSING"]

LATENT_DIM = 16
D_MODEL = 64
NHEAD = 4
N_LAYERS = 2

CONTINUOUS_COLS = [
    "age_census", "cough_d", "fever_d", "wloss_d", "sputum_d", "tbhist_y", "tbtreat_w"
]

BINARY_COLS = [
    "sex_census", "setting", "smoke_now", "smoke_past", "hiv_res",
    "cough", "fever", "weight_loss", "night_sweats", "chest_pain",
    "blood_sputum", "sputum", "hist_rx", "current_rx",
    "xray_normal", "smear_pos", "culture", "cult_pos", "bact"
]

CATEGORICAL_COLS = [
    "region", "married", "edu", "occupation",
    "xrayres", "central_cxr_res",
    "zn", "genexpert", "final_result"
]

ALL_COLS = CONTINUOUS_COLS + BINARY_COLS + CATEGORICAL_COLS


# ============================================================
# SAFE PREPROCESSING
# ============================================================
def build_preprocessor(continuous_cols, binary_cols, categorical_cols):
    cont_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler())
    ])

    bin_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", ohe)
    ])

    return ColumnTransformer([
        ("cont", cont_pipe, continuous_cols),
        ("bin", bin_pipe, binary_cols),
        ("cat", cat_pipe, categorical_cols)
    ])


# ============================================================
# INPUT STANDARDIZATION
# ============================================================
def standardize_missing(df):
    return df.replace(MISSING_MARKERS, np.nan)


def to_numeric_safe(arr):
    import pandas as pd
    return pd.to_numeric(arr, errors="coerce")


def coerce_types(df):
    df = df.copy()

    for c in CONTINUOUS_COLS:
        if c in df.columns:
            df[c] = df[c].replace(MISSING_MARKERS, np.nan)
            df[c] = to_numeric_safe(df[c])

    for c in BINARY_COLS:
        if c in df.columns:
            df[c] = df[c].replace(MISSING_MARKERS, np.nan)
            df[c] = to_numeric_safe(df[c]).clip(0, 1)

    for c in CATEGORICAL_COLS:
        if c in df.columns:
            df[c] = df[c].astype("object")

    return df


def prepare_input_dataframe(df_raw):
    df = df_raw.copy()
    df = standardize_missing(df)

    if "household" in df.columns:
        df = df.drop(columns=["household"])

    for c in CONTINUOUS_COLS:
        if c not in df.columns:
            df[c] = np.nan

    for c in BINARY_COLS:
        if c not in df.columns:
            df[c] = np.nan

    for c in CATEGORICAL_COLS:
        if c not in df.columns:
            df[c] = np.nan

    df = df[ALL_COLS]
    df = coerce_types(df)
    return df


# ============================================================
# LOAD MODEL ARTIFACTS
# ============================================================
def load_feature_names():
    with open("models/feature_names.json", "r") as f:
        return json.load(f)


def load_cluster_model():
    return joblib.load("models/kmeans_model.joblib")


def load_pseudotime_bounds():
    with open("models/pseudotime_bounds.json", "r") as f:
        return json.load(f)


def load_ood_threshold():
    with open("models/ood_threshold.json", "r") as f:
        return json.load(f)


def infer_decoder_structure_from_feature_names(feature_names):
    """
    Exact decoder structure from saved feature_names.json

    Continuous:
      7
    Binary:
      19
    Categorical groups:
      region           -> 4
      married          -> 6
      edu              -> 7
      occupation       -> 10
      xrayres          -> 3
      central_cxr_res  -> 7
      zn               -> 10
      genexpert        -> 2
      final_result     -> 3
    """
    n_cont = 7
    n_bin = 19
    cat_sizes = [4, 6, 7, 10, 3, 7, 10, 2, 3]
    return n_cont, n_bin, cat_sizes


def load_ttvae(input_dim=None):
    feature_names = load_feature_names()
    if input_dim is None:
        input_dim = len(feature_names)

    n_cont, n_bin, cat_sizes = infer_decoder_structure_from_feature_names(feature_names)

    model = TTVAE(
        input_dim=input_dim,
        latent_dim=LATENT_DIM,
        d_model=D_MODEL,
        nhead=NHEAD,
        n_layers=N_LAYERS,
        n_cont=n_cont,
        n_bin=n_bin,
        cat_sizes=cat_sizes
    ).to(device)

    state = torch.load("models/ttvae_best.pth", map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


# ============================================================
# INFERENCE UTILITIES
# ============================================================
def transform_input(df_raw, preprocessor, feature_names):
    import pandas as pd

    df = prepare_input_dataframe(df_raw)
    X = preprocessor.transform(df)

    X_df = pd.DataFrame(X, columns=preprocessor.get_feature_names_out())
    X_df = X_df.reindex(columns=feature_names, fill_value=0.0)

    return df, X_df.values.astype(np.float32)


def compute_latent(model, X):
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    with torch.no_grad():
        mu, _ = model.encode(X_t)
    return mu.cpu().numpy()


def compute_pseudotime(latents, bounds=None):
    pt_raw = PCA(n_components=1, random_state=42).fit_transform(latents).ravel()

    if bounds is None:
        return pt_raw

    pmin = float(bounds["min"])
    pmax = float(bounds["max"])
    pt_norm = (pt_raw - pmin) / (pmax - pmin + 1e-10)
    return np.clip(pt_norm, 0.0, 1.0)


def assign_cluster(kmeans, latents):
    target_dtype = kmeans.cluster_centers_.dtype
    latents = np.asarray(latents, dtype=target_dtype)
    return kmeans.predict(latents)


def batched_reconstruction_error(model, X, batch_size=1024):
    errors = []
    model.eval()

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i + batch_size], dtype=torch.float32, device=device)

            mu, logvar = model.encode(xb)
            z = model.reparameterize(mu, logvar)
            rec = model.decode(z)

            err = ((rec.cpu().numpy() - X[i:i + batch_size]) ** 2).mean(axis=1)
            errors.append(err)

    return np.concatenate(errors)
