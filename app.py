"""
Streamlit App - Prediksi Flexural Bond Strength (τmax) FRP/Steel Bars - Concrete
Berdasarkan: Ebrahimzadeh et al. (2025), Structures 74, 108587
Model: Decision Tree, Random Forest, Gradient Boosting, XGBoost (tuned via Grid Search)
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

# ----------------------------------------------------------------------
# Konfigurasi halaman
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Prediksi Flexural Bond Strength",
    page_icon="🧱",
    layout="wide",
)

DATA_PATH = "Final_Dataset.xlsx"

# Kategori input, urutannya HARUS sama dengan encoding integer di dataset asli
CONCRETE_TYPES = ["NC", "SCC", "FRC", "ECC", "NC-HIGH", "HPC", "UHPC"]
BAR_TYPES = [
    "Steel", "Sand-coated GFRP", "Low modulus GFRP", "Helically GFRP", "BFRP",
    "Plain Steel", "Sand-coated BFRP", "Helically SC GFRP", "CFRP", "HFRP",
    "Epoxy Coated Steel",
]
CONDITION_TYPES = ["Unconditional", "Wet and dry", "Freezing and thawing", "Alkaline solution", "Seawater"]
CONTAMINANT_TYPES = ["Control", "Bond breaker", "Release agent", "Splatter"]

FEATURE_NAMES = [
    "Type of concrete", "fc (MPa)", "Type of bar", "Bar size",
    "Cmin (mm)", "Embedment length (mm)", "Type of Condition", "Contaminant",
]

# Hyperparameter terbaik hasil Grid Search (Table 3 pada paper)
MODEL_BUILDERS = {
    "Decision Tree (DT)": lambda: DecisionTreeRegressor(
        random_state=0, max_depth=9, min_samples_leaf=2,
        min_samples_split=5, max_features="log2",
    ),
    "Random Forest (RF)": lambda: RandomForestRegressor(
        random_state=0, n_estimators=250, max_depth=None,
        min_samples_leaf=1, min_samples_split=2, max_features="sqrt",
    ),
    "Gradient Boosting (GB)": lambda: GradientBoostingRegressor(
        random_state=0, learning_rate=0.15, n_estimators=200, max_depth=3,
    ),
    "XGBoost (XGB)": lambda: XGBRegressor(
        random_state=0, learning_rate=0.45, n_estimators=100, max_depth=3,
    ),
}


# ----------------------------------------------------------------------
# Load data & training model (di-cache supaya cepat)
# ----------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_excel(DATA_PATH, sheet_name=0, header=0)

    # Samakan nama 8 kolom fitur pertama dengan FEATURE_NAMES.
    # Ini WAJIB: nama kolom Excel aslinya sedikit beda dari yang dipakai
    # di kode (mis. "Bar size " ada trailing space, "Embedment lengths (mm)"
    # pakai bentuk jamak). Kalau tidak disamakan, model akan di-fit() dengan
    # nama kolom asli tapi predict() dipanggil dengan nama kolom dari
    # FEATURE_NAMES -> scikit-learn menganggap nama fitur tidak cocok
    # dan melempar ValueError seperti yang muncul di local & Streamlit Cloud.
    rename_map = dict(zip(df.columns[0:8], FEATURE_NAMES))
    df = df.rename(columns=rename_map)

    X = df.iloc[:, 0:8]
    y = df.iloc[:, 8].to_numpy().ravel()
    return df, X, y


@st.cache_resource
def train_all_models(_X, _y):
    """Latih semua model: sekali dengan train/test split (untuk metrik),
    sekali lagi pada seluruh dataset (untuk prediksi final)."""
    Xtr, Xte, ytr, yte = train_test_split(_X, _y, train_size=0.7, random_state=42)

    metrics = {}
    final_models = {}

    for name, builder in MODEL_BUILDERS.items():
        # model untuk evaluasi (dilatih di train set saja)
        eval_model = builder()
        eval_model.fit(Xtr, ytr)
        ypr_tr = eval_model.predict(Xtr)
        ypr_te = eval_model.predict(Xte)

        metrics[name] = {
            "R2_train": round(r2_score(ytr, ypr_tr), 3),
            "R2_test": round(r2_score(yte, ypr_te), 3),
            "RMSE_test": round(mean_squared_error(yte, ypr_te) ** 0.5, 3),
            "MAE_test": round(mean_absolute_error(yte, ypr_te), 3),
        }

        # model final dilatih ulang di seluruh dataset supaya prediksi
        # untuk input baru memakai informasi sebanyak mungkin
        final_model = builder()
        final_model.fit(_X, _y)
        final_models[name] = final_model

    return final_models, metrics, (Xte, yte)


# ----------------------------------------------------------------------
# Sidebar - input fitur
# ----------------------------------------------------------------------
def sidebar_inputs():
    st.sidebar.header("⚙️ Input Fitur")

    concrete = st.sidebar.selectbox("Type of concrete", CONCRETE_TYPES, index=0)
    fc = st.sidebar.number_input("fc (MPa)", min_value=1.0, max_value=250.0, value=30.0, step=1.0)
    bar_type = st.sidebar.selectbox("Type of bar", BAR_TYPES, index=0)
    bar_size = st.sidebar.number_input("Bar size (mm)", min_value=1.0, max_value=40.0, value=12.0, step=1.0)
    cmin = st.sidebar.number_input("Cmin (mm)", min_value=1.0, max_value=100.0, value=40.0, step=1.0)
    embed = st.sidebar.number_input("Embedment length (mm)", min_value=1.0, max_value=500.0, value=120.0, step=5.0)
    condition = st.sidebar.selectbox("Type of Condition", CONDITION_TYPES, index=0)
    contaminant = st.sidebar.selectbox("Contaminant", CONTAMINANT_TYPES, index=0)

    st.sidebar.markdown("---")
    model_name = st.sidebar.radio("Pilih model ML", list(MODEL_BUILDERS.keys()), index=3)

    feature_row = pd.DataFrame(
        [[
            CONCRETE_TYPES.index(concrete),
            fc,
            BAR_TYPES.index(bar_type),
            bar_size,
            cmin,
            embed,
            CONDITION_TYPES.index(condition),
            CONTAMINANT_TYPES.index(contaminant),
        ]],
        columns=FEATURE_NAMES,
    )
    return feature_row, model_name


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    st.title("🧱 Prediksi Flexural Bond Strength (τmax)")
    st.caption(
        "Model machine learning (DT / RF / GB / XGBoost) untuk memprediksi bond strength "
        "antara batang FRP/Steel dan beton."
    )

    df, X, y = load_data()
    final_models, metrics, (Xte, yte) = train_all_models(X, y)

    feature_row, model_name = sidebar_inputs()

    tab_predict, tab_metrics, tab_data = st.tabs(["🔮 Prediksi", "📊 Performa Model", "📁 Dataset"])

    # ---------------- Tab prediksi ----------------
    with tab_predict:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Nilai input yang dipakai")
            st.dataframe(feature_row, use_container_width=True, hide_index=True)

            if st.button("🚀 Prediksi τmax", type="primary", use_container_width=True):
                model = final_models[model_name]
                pred = model.predict(feature_row)[0]
                st.session_state["last_pred"] = pred
                st.session_state["last_model"] = model_name

            if "last_pred" in st.session_state:
                st.metric(
                    label=f"Prediksi τmax ({st.session_state['last_model']})",
                    value=f"{st.session_state['last_pred']:.2f} MPa",
                )

        with col2:
            st.subheader("Prediksi semua model (untuk pembanding)")
            rows = []
            for name, model in final_models.items():
                p = model.predict(feature_row)[0]
                rows.append({"Model": name, "Predicted τmax (MPa)": round(float(p), 2)})
            comp_df = pd.DataFrame(rows)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

            fig, ax = plt.subplots(figsize=(5, 3.2))
            ax.bar(comp_df["Model"], comp_df["Predicted τmax (MPa)"], color="#2E86AB")
            ax.set_ylabel("τmax (MPa)")
            plt.xticks(rotation=20, ha="right")
            fig.tight_layout()
            st.pyplot(fig)

    # ---------------- Tab metrik ----------------
    with tab_metrics:
        st.subheader("Metrik evaluasi (train/test split 70/30, random_state=42)")
        metrics_df = pd.DataFrame(metrics).T
        st.dataframe(metrics_df, use_container_width=True)

        model_for_plot = st.selectbox("Lihat scatter plot Predicted vs Experimental untuk model:", list(final_models.keys()))
        m = final_models[model_for_plot]
        ypr_te = m.predict(Xte)

        fig2, ax2 = plt.subplots(figsize=(5, 5))
        ax2.scatter(yte, ypr_te, s=60, edgecolors="black", facecolors="fuchsia", alpha=0.8)
        lims = [0, max(float(yte.max()), float(ypr_te.max())) + 3]
        ax2.plot(lims, lims, c="black", lw=1.3, label="y = x")
        ax2.set_xlabel("τmax (MPa) - Experimental")
        ax2.set_ylabel("τmax (MPa) - Predicted")
        ax2.set_title(f"{model_for_plot} (Test set)")
        ax2.legend()
        fig2.tight_layout()
        st.pyplot(fig2)

        st.subheader("Feature importance")
        m_full = final_models[model_for_plot]
        if hasattr(m_full, "feature_importances_"):
            imp = pd.Series(m_full.feature_importances_, index=FEATURE_NAMES).sort_values()
            fig3, ax3 = plt.subplots(figsize=(6, 4))
            ax3.barh(imp.index, imp.values, color="#F18F01")
            ax3.set_xlabel("Importance")
            fig3.tight_layout()
            st.pyplot(fig3)
        else:
            st.info("Model ini tidak menyediakan feature_importances_.")

    # ---------------- Tab data ----------------
    with tab_data:
        st.subheader("Dataset yang digunakan")
        st.write(f"Jumlah data: **{df.shape[0]}** baris, **{df.shape[1]}** kolom")
        st.dataframe(df, use_container_width=True)
        st.markdown(
            """
            **Keterangan encoding kategori:**
            - Type of concrete: 0=NC, 1=SCC, 2=FRC, 3=ECC, 4=NC-HIGH, 5=HPC, 6=UHPC
            - Type of bar: 0=Steel, 1=Sand-coated GFRP, 2=Low modulus GFRP, 3=Helically GFRP,
              4=BFRP, 5=Plain Steel, 6=Sand-coated BFRP, 7=Helically SC GFRP, 8=CFRP, 9=HFRP, 10=Epoxy Coated Steel
            - Type of Condition: 0=Unconditional, 1=Wet and dry, 2=Freezing and thawing, 3=Alkaline solution, 4=Seawater
            - Contaminant: 0=Control, 1=Bond breaker, 2=Release agent, 3=Splatter
            """
        )

    st.markdown("---")


if __name__ == "__main__":
    main()
