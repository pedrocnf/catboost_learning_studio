import json
import time
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris, make_classification, make_regression
from sklearn.feature_selection import SelectKBest, mutual_info_classif, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    explained_variance_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, StandardScaler

st.set_page_config(page_title="CatBoost Learning Studio", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    html, body, [class*="css"] {font-size: 125%;}
    .main .block-container {padding-top: 1.15rem; padding-bottom: 2rem; max-width: 96rem;}
    .stButton > button, .stDownloadButton > button {font-size: 1rem !important;}
    [data-testid="stSidebar"] * {font-size: 1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@dataclass
class ExperimentResult:
    experiment_name: str
    timestamp: str
    problem_type: str
    encoding_strategy: str
    feature_selection: str
    n_features: int
    model_params: Dict
    metrics: Dict
    duration_seconds: float
    best_iteration: Optional[int]
    selected_features: List[str]


def init_state() -> None:
    defaults = {
        "df": None,
        "prepared_df": None,
        "dataset_name": None,
        "problem_type": None,
        "target_col": None,
        "feature_cols": None,
        "train_bundle": None,
        "trained_model": None,
        "last_eval": None,
        "experiments": [],
        "eda_saved_charts": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def classify_columns(df: pd.DataFrame, target_col: Optional[str] = None):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
    return numeric_cols, categorical_cols


def infer_problem_type(df: pd.DataFrame, target_col: str) -> str:
    s = df[target_col]
    if pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=True) > 15:
        return "regression"
    return "classification"


def load_demo_dataset(name: str):
    if name == "Iris (classificação)":
        d = load_iris(as_frame=True)
        df = d.frame.copy()
        df.columns = [c.replace(" (cm)", "").replace(" ", "_") for c in df.columns]
        df["target"] = df["target"].map(dict(enumerate(d.target_names)))
        return df, "target", "classification"
    if name == "Breast Cancer (classificação)":
        d = load_breast_cancer(as_frame=True)
        df = d.frame.copy()
        df["target"] = df["target"].map({0: "malignant", 1: "benign"})
        return df, "target", "classification"
    d = load_diabetes(as_frame=True)
    df = d.frame.copy()
    return df, "target", "regression"


def generate_synthetic_classification(
    n_samples: int,
    n_features: int,
    n_informative: int,
    n_redundant: int,
    n_classes: int,
    class_sep: float,
    flip_y: float,
    weights_majority: float,
    missing_pct: float,
    random_seed: int,
):
    weights = None
    if n_classes == 2:
        weights = [weights_majority, 1 - weights_majority]
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_repeated=0,
        n_classes=n_classes,
        class_sep=class_sep,
        flip_y=flip_y,
        weights=weights,
        random_state=random_seed,
    )
    rng = np.random.default_rng(random_seed)
    df = pd.DataFrame(X, columns=[f"num_{i}" for i in range(n_features)])
    df["segmento"] = rng.choice(["A", "B", "C", "D"], len(df), p=[0.40, 0.25, 0.20, 0.15])
    df["canal"] = rng.choice(["online", "loja", "parceiro"], len(df))
    df["perfil"] = rng.choice(["baixo", "médio", "alto"], len(df))
    if n_classes == 2:
        df["target"] = np.where(y == 1, "positivo", "negativo")
    else:
        df["target"] = pd.Series(y).map(lambda v: f"classe_{v}")
    if missing_pct > 0:
        n_missing = int(len(df) * missing_pct)
        for c in ["num_0", "num_1", "segmento"]:
            idx = rng.choice(df.index, size=n_missing, replace=False)
            df.loc[idx, c] = np.nan
    return df, "target", "classification"


def generate_synthetic_regression(
    n_samples: int,
    n_features: int,
    n_informative: int,
    noise: float,
    missing_pct: float,
    random_seed: int,
):
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=noise,
        random_state=random_seed,
    )
    rng = np.random.default_rng(random_seed)
    df = pd.DataFrame(X, columns=[f"num_{i}" for i in range(n_features)])
    df["regiao"] = rng.choice(["Norte", "Sul", "Leste", "Oeste"], len(df))
    df["produto"] = rng.choice(["Basic", "Plus", "Premium"], len(df))
    df["target"] = y
    if missing_pct > 0:
        n_missing = int(len(df) * missing_pct)
        for c in ["num_0", "num_1", "produto"]:
            idx = rng.choice(df.index, size=n_missing, replace=False)
            df.loc[idx, c] = np.nan
    return df, "target", "regression"


def automatic_diagnosis(df: pd.DataFrame, target_col: Optional[str]):
    dtypes = pd.DataFrame({
        "coluna": df.columns,
        "dtype": [str(df[c].dtype) for c in df.columns],
        "missing": [int(df[c].isna().sum()) for c in df.columns],
        "pct_missing": [float(df[c].isna().mean()) for c in df.columns],
        "n_unique": [int(df[c].nunique(dropna=True)) for c in df.columns],
        "amostra": [str(df[c].dropna().iloc[0])[:40] if df[c].dropna().shape[0] else "" for c in df.columns],
    })
    numeric_cols, categorical_cols = classify_columns(df, target_col)
    duplicate_rows = int(df.duplicated().sum())
    diagnostics = {
        "linhas": len(df),
        "colunas": len(df.columns),
        "missing_total": int(df.isna().sum().sum()),
        "duplicadas": duplicate_rows,
        "numéricas": len(numeric_cols),
        "categóricas": len(categorical_cols),
    }
    alerts = []
    high_missing = dtypes.loc[dtypes["pct_missing"] >= 0.30, "coluna"].tolist()
    id_like = [c for c in df.columns if c.lower().startswith("id") or c.lower().endswith("id")]
    constant_like = dtypes.loc[dtypes["n_unique"] <= 1, "coluna"].tolist()
    high_card = dtypes.loc[(dtypes["dtype"] == "object") & (dtypes["n_unique"] > 30), "coluna"].tolist()
    if high_missing:
        alerts.append(f"Colunas com missing ≥ 30%: {high_missing}")
    if id_like:
        alerts.append(f"Possíveis identificadores: {id_like}")
    if constant_like:
        alerts.append(f"Colunas constantes ou quase vazias: {constant_like}")
    if high_card:
        alerts.append(f"Categóricas de alta cardinalidade: {high_card}")
    if target_col and target_col in df.columns and infer_problem_type(df, target_col) == "classification":
        target_dist = df[target_col].astype(str).value_counts(normalize=True, dropna=False)
        if not target_dist.empty and float(target_dist.iloc[0]) > 0.85:
            alerts.append("A variável alvo parece fortemente desbalanceada.")
    return diagnostics, dtypes, alerts


def apply_basic_transformations(df, feature_cols, target_col, missing_num, missing_cat, outlier_clip, log_transform_cols, date_expansion):
    out = df.copy()
    X = out[feature_cols].copy()
    num_cols, cat_cols = classify_columns(X)

    if date_expansion:
        for c in feature_cols:
            if "date" in c.lower() or "data" in c.lower():
                try:
                    dt = pd.to_datetime(X[c], errors="coerce")
                    X[f"{c}_year"] = dt.dt.year
                    X[f"{c}_month"] = dt.dt.month
                    X[f"{c}_day"] = dt.dt.day
                    X[f"{c}_weekday"] = dt.dt.weekday
                except Exception:
                    pass

    for c in log_transform_cols:
        if c in X.columns and pd.api.types.is_numeric_dtype(X[c]):
            series = pd.to_numeric(X[c], errors="coerce")
            shift = max(0, 1 - float(series.min(skipna=True))) if series.notna().any() else 0
            X[c] = np.log1p(series + shift)

    if outlier_clip:
        for c in num_cols:
            s = pd.to_numeric(X[c], errors="coerce")
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            X[c] = s.clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)

    if num_cols and missing_num != "none":
        imp = SimpleImputer(strategy={"mean": "mean", "median": "median", "most_frequent": "most_frequent"}[missing_num])
        X[num_cols] = imp.fit_transform(X[num_cols])
    if cat_cols and missing_cat != "none":
        imp = SimpleImputer(strategy="most_frequent" if missing_cat == "most_frequent" else "constant", fill_value="missing")
        X[cat_cols] = imp.fit_transform(X[cat_cols])

    return pd.concat([X, out[[target_col]]], axis=1)


def apply_encoding(X_train, X_valid, encoding_strategy, scale_numeric):
    num_cols, cat_cols = classify_columns(X_train)
    if encoding_strategy == "catboost_native":
        X_train2, X_valid2 = X_train.copy(), X_valid.copy()
        if scale_numeric != "none" and num_cols:
            scaler = StandardScaler() if scale_numeric == "standard" else MinMaxScaler()
            X_train2[num_cols] = scaler.fit_transform(X_train[num_cols])
            X_valid2[num_cols] = scaler.transform(X_valid[num_cols])
        cat_idx = [X_train2.columns.get_loc(c) for c in cat_cols]
        return X_train2, X_valid2, cat_idx, None

    transformers = []
    if num_cols:
        num_pipe = "passthrough"
        if scale_numeric == "standard":
            num_pipe = Pipeline([("scaler", StandardScaler())])
        elif scale_numeric == "minmax":
            num_pipe = Pipeline([("scaler", MinMaxScaler())])
        transformers.append(("num", num_pipe, num_cols))
    if cat_cols:
        if encoding_strategy == "ordinal":
            cat_pipe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        else:
            cat_pipe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        transformers.append(("cat", cat_pipe, cat_cols))
    ct = ColumnTransformer(transformers=transformers)
    Xtr = ct.fit_transform(X_train)
    Xva = ct.transform(X_valid)
    cols = ct.get_feature_names_out().tolist()
    return pd.DataFrame(Xtr, columns=cols, index=X_train.index), pd.DataFrame(Xva, columns=cols, index=X_valid.index), None, ct


def run_feature_selection(X_train, y_train, problem_type, method, k):
    if method == "none":
        return X_train, list(X_train.columns), pd.DataFrame({"feature": X_train.columns, "score": np.nan})
    k = min(k, X_train.shape[1])
    if method == "variance_filter":
        var = X_train.var(numeric_only=True).sort_values(ascending=False)
        selected = var.head(k).index.tolist() if len(var) else list(X_train.columns[:k])
        return X_train[selected], selected, pd.DataFrame({"feature": var.index, "score": var.values})
    if method == "correlation_filter":
        y_num = pd.factorize(y_train)[0] if problem_type == "classification" else np.asarray(y_train)
        rows = []
        for c in X_train.columns:
            try:
                rows.append((c, abs(np.corrcoef(pd.to_numeric(X_train[c], errors="coerce").fillna(0), y_num)[0, 1])))
            except Exception:
                rows.append((c, 0.0))
        summary = pd.DataFrame(rows, columns=["feature", "score"]).sort_values("score", ascending=False)
        selected = summary.head(k)["feature"].tolist()
        return X_train[selected], selected, summary
    selector = SelectKBest(score_func=mutual_info_classif if problem_type == "classification" else mutual_info_regression, k=k)
    selector.fit(X_train.fillna(0), y_train)
    mask = selector.get_support()
    selected = X_train.columns[mask].tolist()
    summary = pd.DataFrame({"feature": X_train.columns, "score": selector.scores_}).sort_values("score", ascending=False)
    return X_train[selected], selected, summary


def train_catboost(X_train, y_train, X_valid, y_valid, problem_type, cat_features_idx, params):
    model = CatBoostClassifier(**params) if problem_type == "classification" else CatBoostRegressor(**params)
    train_pool = Pool(X_train, label=y_train, cat_features=cat_features_idx)
    valid_pool = Pool(X_valid, label=y_valid, cat_features=cat_features_idx)
    model.fit(train_pool, eval_set=valid_pool, verbose=False, use_best_model=True)
    return model


def evaluate_model(model, X_valid, y_valid, problem_type):
    metrics = {}
    if problem_type == "classification":
        y_pred = pd.Series(model.predict(X_valid)).astype(str)
        y_true = pd.Series(y_valid).astype(str)
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
        metrics["precision_macro"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        metrics["recall_macro"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        metrics["mcc"] = float(matthews_corrcoef(y_true, y_pred))
        try:
            y_proba = model.predict_proba(X_valid)
            classes = [str(c) for c in model.classes_]
            if len(classes) == 2:
                y_true_num = (y_true == classes[1]).astype(int)
                metrics["roc_auc"] = float(roc_auc_score(y_true_num, y_proba[:, 1]))
            metrics["log_loss"] = float(log_loss(y_true, y_proba, labels=classes))
        except Exception:
            pass
    else:
        y_pred = pd.Series(model.predict(X_valid))
        y_true = pd.Series(y_valid)
        mse = float(mean_squared_error(y_true, y_pred))
        metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
        metrics["mse"] = mse
        metrics["rmse"] = float(np.sqrt(mse))
        try:
            metrics["mape"] = float(mean_absolute_percentage_error(y_true.replace(0, np.nan).fillna(1e-9), y_pred))
        except Exception:
            pass
        metrics["r2"] = float(r2_score(y_true, y_pred))
        metrics["explained_variance"] = float(explained_variance_score(y_true, y_pred))
    return metrics


def compute_classic_learning_curve(problem_type, X_train, y_train, X_valid, y_valid, cat_idx, params, random_seed: int = 42):
    fractions = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    rows = []
    X_train = X_train.copy()
    y_train = pd.Series(y_train).copy()

    for frac in fractions:
        n = max(20, int(len(X_train) * frac))
        X_sub = X_train.iloc[:n].copy()
        y_sub = y_train.iloc[:n].copy()

        if problem_type == "classification":
            unique_classes = pd.Series(y_sub).astype(str).nunique()
            if unique_classes < 2:
                continue

        model = train_catboost(X_sub, y_sub, X_valid, y_valid, problem_type, cat_idx, params)

        if problem_type == "classification":
            train_pred = pd.Series(model.predict(X_sub)).astype(str)
            valid_pred = pd.Series(model.predict(X_valid)).astype(str)
            train_score = float(accuracy_score(pd.Series(y_sub).astype(str), train_pred))
            valid_score = float(accuracy_score(pd.Series(y_valid).astype(str), valid_pred))
            metric_name = "Accuracy"
        else:
            train_pred = pd.Series(model.predict(X_sub))
            valid_pred = pd.Series(model.predict(X_valid))
            train_score = float(r2_score(pd.Series(y_sub), train_pred))
            valid_score = float(r2_score(pd.Series(y_valid), valid_pred))
            metric_name = "R2"

        rows.append({
            "train_size": n,
            "fraction": frac,
            "train_score": train_score,
            "validation_score": valid_score,
            "metric_name": metric_name,
        })

    return pd.DataFrame(rows)


def make_eda_chart(df: pd.DataFrame, chart_type: str, x_col: str, y_col: Optional[str], color_col: Optional[str], bins: int):
    title = f"{chart_type}: {x_col}" + (f" vs {y_col}" if y_col else "")
    if chart_type == "Histograma":
        return px.histogram(df, x=x_col, color=color_col if color_col != "Nenhuma" else None, nbins=bins, title=title)
    if chart_type == "Boxplot":
        return px.box(df, x=color_col if color_col != "Nenhuma" else None, y=x_col, title=title)
    if chart_type == "Violin":
        return px.violin(df, x=color_col if color_col != "Nenhuma" else None, y=x_col, box=True, title=title)
    if chart_type == "Barras":
        base = df[x_col].astype(str).value_counts(dropna=False).reset_index()
        base.columns = [x_col, "count"]
        return px.bar(base, x=x_col, y="count", title=title)
    if chart_type == "Scatter":
        return px.scatter(df, x=x_col, y=y_col, color=color_col if color_col != "Nenhuma" else None, title=title)
    if chart_type == "Linha":
        plot_df = df[[x_col, y_col]].dropna().sort_values(x_col)
        return px.line(plot_df, x=x_col, y=y_col, title=title)
    return px.histogram(df, x=x_col, nbins=bins, title=title)


init_state()

st.sidebar.title("🧠 CatBoost Learning Studio")
st.sidebar.caption("Laboratório didático para aprender CatBoost na prática.")
page = st.sidebar.radio(
    "Navegação",
    [
        "1. Visão Geral",
        "2. Dados",
        "3. EDA & Correlação",
        "4. Transformação & Encoding",
        "5. Feature Selection",
        "6. Treinamento",
        "7. Avaliação",
        "8. Interpretabilidade",
        "9. Predição Interativa",
        "10. Comparação de Experimentos",
    ],
)

if page == "1. Visão Geral":
    st.title("🧠 CatBoost Learning Studio")
    st.markdown(
        """
        Este app foi desenhado como um laboratório didático para estudar CatBoost com foco em dados tabulares.

        **Você consegue aqui:**
        - carregar ou gerar datasets
        - escolher explicitamente a variável alvo
        - ver diagnóstico automático dos dados
        - explorar EDA com escolha de gráfico e feature
        - controlar heatmap de correlação
        - treinar, avaliar e comparar experimentos
        """
    )
    # cols = st.columns(4)
    # cards = [
    #     ("Dados", "upload, exemplos e geração sintética"),
    #     ("EDA", "gráficos livres e correlação configurável"),
    #     ("Treinamento", "CatBoost para classificação e regressão"),
    #     ("Avaliação", "métricas, confusion matrix e curvas"),
    # ]
    # for c, (title, text) in zip(cols, cards):
    #     c.metric(title, text)
    st.info("Fluxo recomendado: Dados → EDA → Transformação → Feature Selection → Treinamento → Avaliação.")

elif page == "2. Dados":
    st.title("📦 Dados")
    source = st.radio("Fonte dos dados", ["Dataset de demonstração", "Gerador sintético", "Upload CSV/Excel"], horizontal=True)

    if source == "Dataset de demonstração":
        demo = st.selectbox("Escolha um dataset", ["Iris (classificação)", "Breast Cancer (classificação)", "Diabetes (regressão)"])
        if st.button("Carregar dataset demo"):
            df, target, ptype = load_demo_dataset(demo)
            st.session_state.df = df.copy()
            st.session_state.prepared_df = None
            st.session_state.target_col = target
            st.session_state.problem_type = ptype
            st.session_state.feature_cols = [c for c in df.columns if c != target]
            st.session_state.dataset_name = demo
            st.success(f"Dataset '{demo}' carregado.")


    elif source == "Gerador sintético":
        synthetic_type = st.radio("Tipo de problema sintético", ["classification", "regression"], horizontal=True)
        col_a, col_b, col_c, col_d = st.columns(4)
        n_samples = col_a.slider("Amostras", 200, 5000, 1200, 100)
        n_features = col_b.slider("Features numéricas", 4, 30, 12, 1)
        random_seed = col_c.number_input("Seed", 0, 99999, 42)
        missing_pct = col_d.slider("Missing artificial (%)", 0.0, 0.30, 0.08, 0.01)

        if synthetic_type == "classification":
            col1, col2, col3, col4 = st.columns(4)
            n_informative = col1.slider("Informativas", 2, n_features, min(8, n_features), 1)
            n_redundant = col2.slider("Redundantes", 0, max(0, n_features - n_informative), min(2, max(0, n_features - n_informative)), 1)
            n_classes = col3.slider("Classes", 2, 5, 2, 1)
            class_sep = col4.slider("Separação entre classes", 0.2, 3.0, 1.0, 0.1)
            col5, col6 = st.columns(2)
            flip_y = col5.slider("Ruído de rótulo (flip_y)", 0.0, 0.30, 0.03, 0.01)
            weights_majority = col6.slider("Proporção da classe majoritária", 0.50, 0.95, 0.72, 0.01, disabled=(n_classes != 2))
            if st.button("Gerar dataset sintético"):
                df, target, ptype = generate_synthetic_classification(
                    n_samples, n_features, n_informative, n_redundant, n_classes, class_sep, flip_y,
                    weights_majority if n_classes == 2 else 0.5, missing_pct, int(random_seed)
                )
                st.session_state.df = df.copy()
                st.session_state.prepared_df = None
                st.session_state.target_col = target
                st.session_state.problem_type = ptype
                st.session_state.feature_cols = [c for c in df.columns if c != target]
                st.session_state.dataset_name = "Sintético Classificação"
                st.success("Dataset sintético de classificação gerado.")
        else:
            col1, col2 = st.columns(2)
            n_informative = col1.slider("Informativas", 2, n_features, min(8, n_features), 1)
            noise = col2.slider("Ruído", 0.0, 100.0, 18.0, 1.0)
            if st.button("Gerar dataset sintético"):
                df, target, ptype = generate_synthetic_regression(n_samples, n_features, n_informative, noise, missing_pct, int(random_seed))
                st.session_state.df = df.copy()
                st.session_state.prepared_df = None
                st.session_state.target_col = target
                st.session_state.problem_type = ptype
                st.session_state.feature_cols = [c for c in df.columns if c != target]
                st.session_state.dataset_name = "Sintético Regressão"
                st.success("Dataset sintético de regressão gerado.")

    else:
        file = st.file_uploader("Envie um CSV ou Excel", type=["csv", "xlsx"])
        if file is not None:
            df = pd.read_csv(file) if file.name.lower().endswith(".csv") else pd.read_excel(file)
            st.session_state.df = df.copy()
            st.session_state.prepared_df = None
            st.session_state.dataset_name = file.name
            st.success(f"Arquivo '{file.name}' carregado.")

    df = st.session_state.df
    if df is not None:
        st.subheader("Pré-visualização")
        st.dataframe(df.head(20), use_container_width=True)

        st.subheader("Configuração explícita da variável alvo")
        col1, col2 = st.columns([2, 1])
        target_default = st.session_state.target_col if st.session_state.target_col in df.columns else df.columns[-1]
        target = col1.selectbox("Variável de decisão / target", options=df.columns.tolist(), index=df.columns.tolist().index(target_default))
        inferred = infer_problem_type(df, target)
        problem_type = col2.radio("Tipo de problema", ["classification", "regression"], index=0 if inferred == "classification" else 1, horizontal=True)
        feature_cols = st.multiselect("Features a utilizar", options=[c for c in df.columns if c != target], default=[c for c in df.columns if c != target])
        if st.button("Salvar configuração atual"):
            st.session_state.target_col = target
            st.session_state.problem_type = problem_type
            st.session_state.feature_cols = feature_cols
            st.success("Configuração salva.")

        diagnostics, dtype_df, alerts = automatic_diagnosis(df, target)
        st.subheader("Diagnóstico automático dos dados")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Linhas", diagnostics["linhas"])
        c2.metric("Colunas", diagnostics["colunas"])
        c3.metric("Missing", diagnostics["missing_total"])
        c4.metric("Duplicadas", diagnostics["duplicadas"])
        c5.metric("Numéricas", diagnostics["numéricas"])
        c6.metric("Categóricas", diagnostics["categóricas"])
        if alerts:
            for alert in alerts:
                st.warning(alert)
        else:
            st.success("Nenhum alerta estrutural importante foi detectado pelas regras automáticas básicas.")
        st.dataframe(dtype_df, use_container_width=True)

        if target in df.columns:
            st.subheader("Distribuição da variável alvo")
            if problem_type == "classification":
                target_df = df[target].astype(str).value_counts(dropna=False).reset_index()
                target_df.columns = [target, "count"]
                st.plotly_chart(px.bar(target_df, x=target, y="count", title="Distribuição do target"), use_container_width=True)
            else:
                st.plotly_chart(px.histogram(df, x=target, nbins=40, title="Distribuição do target"), use_container_width=True)
    else:
        st.info("Carregue ou gere um dataset para continuar.")

elif page == "3. EDA & Correlação":
    st.title("📊 EDA & Correlação")
    df = st.session_state.prepared_df if st.session_state.prepared_df is not None else st.session_state.df
    target = st.session_state.target_col
    if df is None:
        st.warning("Primeiro carregue os dados.")
    else:
        features = [c for c in df.columns if c != target]
        num_cols, cat_cols = classify_columns(df, target)

        st.subheader("Construtor de gráficos")
        col1, col2, col3, col4 = st.columns(4)
        chart_type = col1.selectbox("Tipo de gráfico", ["Histograma", "Boxplot", "Violin", "Barras", "Scatter", "Linha"])
        x_candidates = df.columns.tolist()
        x_col = col2.selectbox("Feature principal", x_candidates)
        scatter_line_requires_y = chart_type in ["Scatter", "Linha"]
        y_candidates = [c for c in df.columns if c != x_col]
        y_col = col3.selectbox("Segunda feature", y_candidates, disabled=not scatter_line_requires_y) if y_candidates else None
        color_options = ["Nenhuma"] + [c for c in cat_cols if c != x_col and c != y_col]
        color_col = col4.selectbox("Agrupar por cor", color_options)
        bins = st.slider("Bins do histograma", 5, 100, 30)

        fig = make_eda_chart(df, chart_type, x_col, y_col if scatter_line_requires_y else None, color_col, bins)
        st.plotly_chart(fig, use_container_width=True)

        add_col, clear_col = st.columns(2)
        if add_col.button("Adicionar gráfico ao painel"):
            st.session_state.eda_saved_charts.append({
                "chart_type": chart_type,
                "x_col": x_col,
                "y_col": y_col if scatter_line_requires_y else None,
                "color_col": color_col,
                "bins": bins,
            })
            st.success("Gráfico adicionado ao painel.")
        if clear_col.button("Limpar painel salvo"):
            st.session_state.eda_saved_charts = []
            st.success("Painel limpo.")

        if st.session_state.eda_saved_charts:
            st.subheader("Painel de gráficos salvos")
            for i, cfg in enumerate(st.session_state.eda_saved_charts):
                top1, top2 = st.columns([5, 1])
                with top1:
                    st.plotly_chart(make_eda_chart(df, cfg["chart_type"], cfg["x_col"], cfg.get("y_col"), cfg.get("color_col", "Nenhuma"), cfg.get("bins", 30)), use_container_width=True)
                with top2:
                    if st.button(f"Remover #{i+1}"):
                        st.session_state.eda_saved_charts.pop(i)
                        st.rerun()

        st.subheader("Heatmap de correlação")
        if not num_cols:
            st.info("Não há colunas numéricas suficientes para correlação.")
        else:
            selected_heatmap_cols = st.multiselect(
                "Selecione as colunas numéricas para o heatmap",
                options=num_cols,
                default=num_cols[: min(8, len(num_cols))],
            )
            if len(selected_heatmap_cols) >= 2:
                corr = df[selected_heatmap_cols].corr(numeric_only=True)
                fig_corr = px.imshow(corr, text_auto=".2f", aspect="auto", title="Heatmap de correlação")
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.info("Escolha pelo menos 2 variáveis numéricas para gerar o heatmap.")

elif page == "4. Transformação & Encoding":
    st.title("🛠️ Transformação & Encoding")
    df = st.session_state.df
    target = st.session_state.target_col
    features = st.session_state.feature_cols
    if df is None or target is None or not features:
        st.warning("Primeiro configure os dados.")
    else:
        num_cols, _ = classify_columns(df[features + [target]], target)
        c1, c2 = st.columns(2)
        with c1:
            missing_num = st.selectbox("Imputação numérica", ["none", "mean", "median", "most_frequent"], index=2)
            missing_cat = st.selectbox("Imputação categórica", ["none", "most_frequent", "constant"], index=2)
            outlier_clip = st.checkbox("Clipping de outliers (IQR)", value=False)
            date_expansion = st.checkbox("Expandir colunas de data", value=True)
        with c2:
            log_transform_cols = st.multiselect("Aplicar log1p em colunas", options=num_cols, default=[])
            encoding_strategy = st.selectbox("Estratégia de encoding", ["catboost_native", "onehot", "ordinal"], index=0)
            scale_numeric = st.selectbox("Escalonamento numérico", ["none", "standard", "minmax"], index=0)
        if st.button("Aplicar transformações"):
            transformed = apply_basic_transformations(df, features, target, missing_num, missing_cat, outlier_clip, log_transform_cols, date_expansion)
            st.session_state.prepared_df = transformed
            st.session_state.train_bundle = {
                "encoding_strategy": encoding_strategy,
                "scale_numeric": scale_numeric,
            }
            st.success("Transformações aplicadas.")
        base_df = st.session_state.prepared_df if st.session_state.prepared_df is not None else df[features + [target]].copy()
        st.dataframe(base_df.head(20), use_container_width=True)

elif page == "5. Feature Selection":
    st.title("🧬 Feature Selection")
    df = st.session_state.prepared_df if st.session_state.prepared_df is not None else st.session_state.df
    target = st.session_state.target_col
    problem_type = st.session_state.problem_type
    if df is None or target is None:
        st.warning("Primeiro carregue os dados.")
    else:
        feature_cols = [c for c in df.columns if c != target]
        X = df[feature_cols].copy()
        y = df[target].copy()
        method = st.selectbox("Método", ["none", "variance_filter", "correlation_filter", "selectkbest"], index=0)
        k = st.slider("Número máximo de features", 1, max(1, len(feature_cols)), min(10, len(feature_cols)))
        preview_df = X.copy()
        for c in preview_df.select_dtypes(exclude=[np.number]).columns.tolist():
            preview_df[c] = pd.factorize(preview_df[c].astype(str))[0]
        selected_X, selected_features, summary = run_feature_selection(preview_df, y, problem_type, method, k)
        st.write(selected_features)
        st.dataframe(summary.head(50), use_container_width=True)
        if st.button("Salvar features selecionadas"):
            current = st.session_state.prepared_df if st.session_state.prepared_df is not None else st.session_state.df
            kept = [c for c in current.columns if c in selected_features or c == target]
            st.session_state.prepared_df = current[kept].copy()
            st.session_state.feature_cols = [c for c in kept if c != target]
            st.success("Features salvas.")

elif page == "6. Treinamento":
    st.title("🚀 Treinamento")
    df = st.session_state.prepared_df if st.session_state.prepared_df is not None else st.session_state.df
    target = st.session_state.target_col
    problem_type = st.session_state.problem_type
    bundle = st.session_state.train_bundle or {"encoding_strategy": "catboost_native", "scale_numeric": "none"}
    if df is None or target is None:
        st.warning("Primeiro configure os dados.")
    else:
        feature_cols = [c for c in df.columns if c != target]
        X = df[feature_cols].copy()
        y = df[target].copy()
        c1, c2, c3 = st.columns(3)
        test_size = c1.slider("Tamanho da validação", 0.1, 0.4, 0.2, 0.05)
        random_seed = c2.number_input("Random seed", 0, 99999, 42)
        stratify_flag = c3.checkbox("Stratify", value=(problem_type == "classification"))
        c1, c2, c3 = st.columns(3)
        iterations = c1.slider("iterations", 50, 2000, 300, 50)
        learning_rate = c2.slider("learning_rate", 0.01, 0.50, 0.08, 0.01)
        depth = c3.slider("depth", 2, 12, 6)
        c4, c5, c6 = st.columns(3)
        l2_leaf_reg = c4.slider("l2_leaf_reg", 1.0, 20.0, 3.0, 0.5)
        bagging_temperature = c5.slider("bagging_temperature", 0.0, 10.0, 1.0, 0.1)
        border_count = c6.slider("border_count", 16, 255, 128)
        c7, c8 = st.columns(2)
        early_stopping_rounds = c7.slider("early_stopping_rounds", 10, 200, 40, 5)
        auto_class_weights = c8.selectbox("class_weights", ["None", "Balanced"], index=0 if problem_type != "classification" else 1)
        encoding_strategy = st.selectbox("Encoding para o treino", ["catboost_native", "onehot", "ordinal"], index=["catboost_native", "onehot", "ordinal"].index(bundle.get("encoding_strategy", "catboost_native")))
        scale_numeric = st.selectbox("Scaling numérico", ["none", "standard", "minmax"], index=["none", "standard", "minmax"].index(bundle.get("scale_numeric", "none")))
        experiment_name = st.text_input("Nome do experimento", value=f"baseline_{int(time.time())}")
        if st.button("Treinar modelo"):
            progress = st.progress(0)
            status = st.empty()
            start = time.perf_counter()
            status.info("Etapa 1/5: split dos dados")
            progress.progress(15)
            stratify = y if (problem_type == "classification" and stratify_flag) else None
            X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=test_size, random_state=int(random_seed), stratify=stratify)
            status.info("Etapa 2/5: encoding e scaling")
            progress.progress(35)
            X_train_enc, X_valid_enc, cat_idx, preprocessor = apply_encoding(X_train, X_valid, encoding_strategy, scale_numeric)
            status.info("Etapa 3/5: configuração do modelo")
            progress.progress(55)
            params = {
                "iterations": int(iterations),
                "learning_rate": float(learning_rate),
                "depth": int(depth),
                "l2_leaf_reg": float(l2_leaf_reg),
                "bagging_temperature": float(bagging_temperature),
                "border_count": int(border_count),
                "random_seed": int(random_seed),
                "early_stopping_rounds": int(early_stopping_rounds),
                "loss_function": "Logloss" if problem_type == "classification" else "RMSE",
                "eval_metric": "Logloss" if problem_type == "classification" else "RMSE",
                "custom_metric": ["Accuracy", "AUC"] if problem_type == "classification" else ["MAE", "R2"],
                "auto_class_weights": None if auto_class_weights == "None" else auto_class_weights,
            }
            status.info("Etapa 4/5: treinamento")
            progress.progress(80)
            model = train_catboost(X_train_enc, y_train, X_valid_enc, y_valid, problem_type, cat_idx, params)
            status.info("Etapa 5/5: avaliação")
            metrics = evaluate_model(model, X_valid_enc, y_valid, problem_type)
            progress.progress(100)
            status.success("Treinamento concluído.")
            duration = time.perf_counter() - start
            st.session_state.trained_model = model
            st.session_state.last_eval = {
                "X_valid": X_valid_enc,
                "y_valid": y_valid,
                "feature_names": X_train_enc.columns.tolist(),
                "preprocessor": preprocessor,
                "original_feature_names": feature_cols,
                "encoding_strategy": encoding_strategy,
                "scale_numeric": scale_numeric,
                "evals_result": model.get_evals_result(),
                "X_train": X_train_enc,
                "y_train": y_train,
                "cat_idx": cat_idx,
                "params": params,
            }
            exp = ExperimentResult(
                experiment_name=experiment_name,
                timestamp=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                problem_type=problem_type,
                encoding_strategy=encoding_strategy,
                feature_selection="saved_current_features",
                n_features=X_train_enc.shape[1],
                model_params=params,
                metrics=metrics,
                duration_seconds=duration,
                best_iteration=getattr(model, "get_best_iteration", lambda: None)(),
                selected_features=X_train_enc.columns.tolist(),
            )
            st.session_state.experiments.append(asdict(exp))
            st.dataframe(pd.DataFrame([metrics]), use_container_width=True)

elif page == "7. Avaliação":
    st.title("📈 Avaliação")
    model = st.session_state.trained_model
    bundle = st.session_state.last_eval
    problem_type = st.session_state.problem_type
    if model is None or bundle is None:
        st.warning("Treine um modelo primeiro.")
    else:
        X_valid = bundle["X_valid"]
        y_valid = pd.Series(bundle["y_valid"])
        metrics = evaluate_model(model, X_valid, y_valid, problem_type)
        cols = st.columns(min(4, max(1, len(metrics))))
        for i, (k, v) in enumerate(metrics.items()):
            cols[i % len(cols)].metric(k, f"{v:.4f}")

        evals_result = bundle.get("evals_result") or getattr(model, "get_evals_result", lambda: {})()
        if evals_result:
            st.subheader("Learning Curve")
            dataset_names = list(evals_result.keys())
            curve_frames = []
            for dataset_name in dataset_names:
                metric_map = evals_result.get(dataset_name, {})
                preferred_metric = None
                if problem_type == "classification":
                    preferred_metric = "Logloss" if "Logloss" in metric_map else next(iter(metric_map), None)
                else:
                    preferred_metric = "RMSE" if "RMSE" in metric_map else next(iter(metric_map), None)
                if preferred_metric is not None:
                    vals = metric_map.get(preferred_metric, [])
                    curve_frames.append(pd.DataFrame({
                        "iteration": list(range(len(vals))),
                        "value": vals,
                        "dataset": dataset_name,
                        "metric": preferred_metric,
                    }))
            if curve_frames:
                learning_df = pd.concat(curve_frames, ignore_index=True)
                fig_learning = px.line(
                    learning_df,
                    x="iteration",
                    y="value",
                    color="dataset",
                    markers=True,
                    title="Learning Curve",
                    labels={"value": "Valor da métrica", "iteration": "Iteração"},
                )
                st.plotly_chart(fig_learning, use_container_width=True)

            if problem_type == "classification":
                acc_frames = []
                for dataset_name in dataset_names:
                    metric_map = evals_result.get(dataset_name, {})
                    if "Accuracy" in metric_map:
                        vals = metric_map["Accuracy"]
                        acc_frames.append(pd.DataFrame({
                            "iteration": list(range(len(vals))),
                            "accuracy": vals,
                            "dataset": dataset_name,
                        }))
                if acc_frames:
                    st.subheader("Curva de Acurácia")
                    acc_df = pd.concat(acc_frames, ignore_index=True)
                    fig_acc = px.line(
                        acc_df,
                        x="iteration",
                        y="accuracy",
                        color="dataset",
                        markers=True,
                        title="Curva de Acurácia",
                        labels={"accuracy": "Acurácia", "iteration": "Iteração"},
                    )
                    st.plotly_chart(fig_acc, use_container_width=True)

        if bundle.get("X_train") is not None and bundle.get("params") is not None:
            st.subheader("Learning Curve Clássica")
            try:
                lc_df = compute_classic_learning_curve(
                    problem_type=problem_type,
                    X_train=bundle["X_train"],
                    y_train=bundle["y_train"],
                    X_valid=bundle["X_valid"],
                    y_valid=bundle["y_valid"],
                    cat_idx=bundle.get("cat_idx"),
                    params=bundle["params"],
                    random_seed=int(bundle["params"].get("random_seed", 42)),
                )
                if not lc_df.empty:
                    long_df = pd.concat([
                        lc_df[["train_size", "metric_name", "train_score"]].rename(columns={"train_score": "score"}).assign(curva="Treino"),
                        lc_df[["train_size", "metric_name", "validation_score"]].rename(columns={"validation_score": "score"}).assign(curva="Validação"),
                    ], ignore_index=True)
                    fig_lc = px.line(
                        long_df,
                        x="train_size",
                        y="score",
                        color="curva",
                        markers=True,
                        title="Learning Curve Clássica",
                        labels={"train_size": "Tamanho do conjunto de treino", "score": lc_df["metric_name"].iloc[0]},
                    )
                    st.plotly_chart(fig_lc, use_container_width=True)
                    st.caption("Nesta curva clássica, a linha de treino tende a começar mais alta e cair, enquanto a validação tende a subir e se aproximar, ajudando a visualizar overfitting e ganho de generalização.")
                    st.dataframe(lc_df, use_container_width=True)
            except Exception as e:
                st.info(f"Não foi possível calcular a learning curve clássica: {e}")

        if problem_type == "classification":
            y_pred = pd.Series(model.predict(X_valid)).astype(str)
            y_true = y_valid.astype(str)
            labels = sorted(list(set(y_true.unique()).union(set(y_pred.unique()))))
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            cm_df = pd.DataFrame(cm, index=[f"real_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])
            st.dataframe(cm_df, use_container_width=True)
            st.plotly_chart(px.imshow(cm_df, text_auto=True, aspect="auto", title="Matriz de confusão"), use_container_width=True)
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            st.dataframe(pd.DataFrame(report).T, use_container_width=True)
            try:
                proba = model.predict_proba(X_valid)
                classes = [str(c) for c in model.classes_]
                if len(classes) == 2:
                    pos_label = classes[1]
                    y_bin = (y_true == pos_label).astype(int)
                    scores = proba[:, 1]
                    fpr, tpr, _ = roc_curve(y_bin, scores)
                    roc_auc = roc_auc_score(y_bin, scores)
                    st.plotly_chart(px.line(pd.DataFrame({"fpr": fpr, "tpr": tpr}), x="fpr", y="tpr", title=f"ROC Curve (AUC={roc_auc:.4f})"), use_container_width=True)
                    prec, rec, _ = precision_recall_curve(y_bin, scores)
                    st.plotly_chart(px.line(pd.DataFrame({"precision": prec, "recall": rec}), x="recall", y="precision", title="Precision-Recall Curve"), use_container_width=True)
            except Exception:
                pass
        else:
            y_pred = pd.Series(model.predict(X_valid))
            reg_df = pd.DataFrame({"real": y_valid, "pred": y_pred})
            st.plotly_chart(px.scatter(reg_df, x="real", y="pred", trendline="ols", title="Predito vs Real"), use_container_width=True)

elif page == "8. Interpretabilidade":
    st.title("🔎 Interpretabilidade")
    model = st.session_state.trained_model
    bundle = st.session_state.last_eval
    if model is None or bundle is None:
        st.warning("Treine um modelo primeiro.")
    else:
        X_valid = bundle["X_valid"]
        y_valid = bundle["y_valid"]
        feature_names = bundle["feature_names"]
        fi = model.get_feature_importance()
        fi_df = pd.DataFrame({"feature": feature_names, "importance": fi}).sort_values("importance", ascending=False)
        st.dataframe(fi_df, use_container_width=True)
        st.plotly_chart(px.bar(fi_df.head(25), x="feature", y="importance", title="Feature Importance"), use_container_width=True)
        try:
            perm = permutation_importance(model, X_valid, y_valid, n_repeats=5, random_state=42)
            perm_df = pd.DataFrame({"feature": feature_names, "importance_mean": perm.importances_mean}).sort_values("importance_mean", ascending=False)
            st.dataframe(perm_df, use_container_width=True)
        except Exception:
            pass

elif page == "9. Predição Interativa":
    st.title("🎯 Predição Interativa")
    model = st.session_state.trained_model
    df = st.session_state.prepared_df if st.session_state.prepared_df is not None else st.session_state.df
    target = st.session_state.target_col
    bundle = st.session_state.last_eval
    if model is None or df is None or target is None or bundle is None:
        st.warning("Treine um modelo primeiro.")
    else:
        orig_features = [c for c in df.columns if c != target]
        form_data = {}
        cols = st.columns(2)
        for i, c in enumerate(orig_features):
            s = df[c]
            with cols[i % 2]:
                if pd.api.types.is_numeric_dtype(s):
                    form_data[c] = st.number_input(c, value=float(s.dropna().median()) if s.dropna().shape[0] else 0.0)
                else:
                    options = s.dropna().astype(str).unique().tolist() or ["missing"]
                    form_data[c] = st.selectbox(c, options)
        if st.button("Prever"):
            input_df = pd.DataFrame([form_data])[bundle["original_feature_names"]]
            if bundle["encoding_strategy"] == "catboost_native":
                row_for_pred = input_df.copy()
            else:
                arr = bundle["preprocessor"].transform(input_df)
                row_for_pred = pd.DataFrame(arr, columns=bundle["preprocessor"].get_feature_names_out().tolist())
            pred = model.predict(row_for_pred)
            st.success(f"Predição: {pred[0] if hasattr(pred, '__len__') else pred}")

else:
    st.title("🧪 Comparação de Experimentos")
    experiments = st.session_state.experiments
    if not experiments:
        st.warning("Ainda não há experimentos salvos.")
    else:
        exp_df = pd.DataFrame(experiments)
        metrics_df = pd.json_normalize(exp_df["metrics"])
        params_df = pd.json_normalize(exp_df["model_params"])
        final_df = pd.concat([exp_df.drop(columns=["metrics", "model_params", "selected_features"]), metrics_df.add_prefix("metric_"), params_df.add_prefix("param_")], axis=1)
        st.dataframe(final_df, use_container_width=True)
        metric_cols = [c for c in final_df.columns if c.startswith("metric_")]
        if metric_cols:
            y_metric = st.selectbox("Métrica para comparar", metric_cols)
            st.plotly_chart(px.bar(final_df, x="experiment_name", y=y_metric, color="encoding_strategy", title=f"Comparação por {y_metric}"), use_container_width=True)

st.markdown("<div style='font-size:0.72rem; color:#6b7280; margin-top:1.5rem;'>Prof. Pedro Nascimento • GitHub: www.github.com/pedrocnf</div>", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.caption("Execução: streamlit run app.py")
st.sidebar.caption("Prof. Pedro Nascimento • www.github.com/pedrocnf")
