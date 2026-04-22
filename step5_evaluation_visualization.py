"""
==============================================================================
STEP 5: EVALUASI, VISUALISASI & ANALISIS REGRESI
Title: Transforming User Feedback into Strategic Intelligence:
       An ABSA of balé by BTN Superapp using IndoBERT and E-S-QUAL Framework
==============================================================================
Tujuan  :
  (a) Evaluasi komprehensif model dengan metrik per-kelas
  (b) Visualisasi distribusi sentimen per dimensi E-S-QUAL
  (c) Analisis regresi Ridge: pengaruh aspek sentimen terhadap rating
  (d) Ekspor hasil untuk penulisan jurnal

Input   :
  - models/best_model.pt
  - data/labeled_reviews.csv
  - results/training_metrics.json

Output  :
  - results/confusion_matrix_*.png
  - results/sentiment_distribution.png
  - results/regression_analysis.csv
  - results/final_report.json

Dependensi:
    pip install matplotlib seaborn pandas numpy scikit-learn scipy torch
    pip install transformers shap wordcloud
==============================================================================
"""

import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from typing import Dict, List

from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/step5_evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────
LABELED_PATH  = "data/labeled_reviews.csv"
METRICS_PATH  = "results/training_metrics.json"
MODEL_PATH    = "models/best_model.pt"
RESULT_DIR    = "results/"
FIGURE_DIR    = "results/figures/"

ASPECTS    = ["EFFICIENCY", "SYSTEM_AVAILABILITY", "FULFILLMENT", "PRIVACY"]
SENTIMENTS = ["Positif", "Netral", "Negatif"]
COLORS     = {
    "Positif": "#2ecc71",
    "Netral" : "#f39c12",
    "Negatif": "#e74c3c",
}
BTN_COLORS = ["#003087", "#006DC6", "#0091DA", "#00ADEF"]  # Palet warna BTN

Path(FIGURE_DIR).mkdir(parents=True, exist_ok=True)
Path(RESULT_DIR).mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# 1. CONFUSION MATRIX PER ASPEK
# ─────────────────────────────────────────────
def plot_confusion_matrices(
    cm_data: Dict[str, np.ndarray],
    labels: List[str] = SENTIMENTS,
    output_dir: str = FIGURE_DIR
) -> None:
    """
    Plot confusion matrix untuk setiap aspek E-S-QUAL.
    Menggunakan normalisasi per-baris untuk menampilkan recall.
    """
    n_aspects = len(cm_data)
    fig, axes = plt.subplots(1, n_aspects, figsize=(5 * n_aspects, 5))
    fig.suptitle(
        "Confusion Matrix per Dimensi E-S-QUAL\n(IndoBERT ABSA — balé by BTN)",
        fontsize=14, fontweight="bold", y=1.02
    )

    for ax, (aspect, cm) in zip(axes, cm_data.items()):
        # Normalisasi per baris (recall)
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            cbar=True,
            linewidths=0.5,
        )
        ax.set_title(f"{aspect}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Prediksi", fontsize=9)
        ax.set_ylabel("Aktual",   fontsize=9)
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    out_path = Path(output_dir) / "confusion_matrices.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrix disimpan: {out_path}")


# ─────────────────────────────────────────────
# 2. DISTRIBUSI SENTIMEN PER DIMENSI E-S-QUAL
# ─────────────────────────────────────────────
def plot_sentiment_distribution(
    df: pd.DataFrame,
    aspects: List[str] = ASPECTS,
    output_dir: str = FIGURE_DIR
) -> None:
    """
    Bar chart bertumpuk (stacked) menampilkan proporsi sentimen
    Positif/Netral/Negatif untuk setiap dimensi E-S-QUAL.
    """
    records = []
    for asp in aspects:
        mask   = df["aspects"].str.contains(asp, na=False)
        subset = df[mask]["sentiment_rule"].value_counts()
        total  = subset.sum()
        for sent in SENTIMENTS:
            count = subset.get(sent, 0)
            records.append({
                "Aspek"    : asp.replace("_", " ").title(),
                "Sentimen" : sent,
                "Count"    : count,
                "Pct"      : count / total * 100 if total > 0 else 0,
            })

    dist_df = pd.DataFrame(records)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Distribusi Sentimen per Dimensi E-S-QUAL\nbalé by BTN Superapp",
        fontsize=13, fontweight="bold"
    )

    # Plot 1: Count absolut
    pivot_count = dist_df.pivot(index="Aspek", columns="Sentimen", values="Count").fillna(0)
    pivot_count[SENTIMENTS].plot(
        kind="bar",
        ax=ax1,
        color=[COLORS[s] for s in SENTIMENTS],
        edgecolor="white",
        width=0.7,
    )
    ax1.set_title("Jumlah Ulasan per Aspek dan Sentimen", fontsize=11)
    ax1.set_xlabel("Dimensi E-S-QUAL", fontsize=10)
    ax1.set_ylabel("Jumlah Ulasan", fontsize=10)
    ax1.legend(title="Sentimen", fontsize=9)
    ax1.tick_params(axis="x", rotation=30)

    # Plot 2: Persentase (stacked)
    pivot_pct = dist_df.pivot(index="Aspek", columns="Sentimen", values="Pct").fillna(0)
    pivot_pct[SENTIMENTS].plot(
        kind="bar",
        stacked=True,
        ax=ax2,
        color=[COLORS[s] for s in SENTIMENTS],
        edgecolor="white",
        width=0.7,
    )
    ax2.set_title("Proporsi Sentimen per Aspek (%)", fontsize=11)
    ax2.set_xlabel("Dimensi E-S-QUAL", fontsize=10)
    ax2.set_ylabel("Persentase (%)", fontsize=10)
    ax2.legend(title="Sentimen", fontsize=9, loc="upper right")
    ax2.tick_params(axis="x", rotation=30)

    # Tambahkan garis referensi 50%
    ax2.axhline(y=50, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)

    plt.tight_layout()
    out_path = Path(output_dir) / "sentiment_distribution.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Distribusi sentimen disimpan: {out_path}")

    return dist_df


# ─────────────────────────────────────────────
# 3. WORDCLOUD PER ASPEK
# ─────────────────────────────────────────────
def plot_wordclouds(
    df: pd.DataFrame,
    aspects: List[str] = ASPECTS,
    output_dir: str = FIGURE_DIR
) -> None:
    """WordCloud per aspek untuk insight visual kata kunci."""
    try:
        from wordcloud import WordCloud
        import matplotlib.colors as mcolors

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        fig.suptitle(
            "Word Cloud per Dimensi E-S-QUAL — balé by BTN",
            fontsize=13, fontweight="bold"
        )

        for ax, asp, color in zip(axes, aspects, BTN_COLORS):
            mask   = df["aspects"].str.contains(asp, na=False)
            texts  = " ".join(df[mask]["clean_text"].dropna().tolist())

            if not texts.strip():
                ax.set_visible(False)
                continue

            def color_func(*args, **kwargs):
                return color

            wc = WordCloud(
                width=600, height=400,
                background_color="white",
                color_func=color_func,
                max_words=80,
                font_path=None,         # Ganti dengan path font Indonesia jika ada
                collocations=False,
            ).generate(texts)

            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            ax.set_title(asp.replace("_", " ").title(), fontsize=11, fontweight="bold")

        plt.tight_layout()
        out_path = Path(output_dir) / "wordclouds.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"WordCloud disimpan: {out_path}")
    except ImportError:
        logger.warning("wordcloud tidak tersedia. Plot dilewati.")


# ─────────────────────────────────────────────
# 4. ANALISIS REGRESI RIDGE
# ─────────────────────────────────────────────
def build_sentiment_features(
    df: pd.DataFrame,
    aspects: List[str] = ASPECTS,
) -> pd.DataFrame:
    """
    Membangun feature matrix untuk regresi:
    - Variabel independen (X): skor sentimen per dimensi E-S-QUAL
      (Positif=1, Netral=0, Negatif=-1)
    - Variabel dependen (y): rating bintang (1-5)

    Skor sentimen dibuat sebagai rata-rata per aspek dari ulasan yang
    mengandung aspek tersebut.
    """
    sent_score_map = {"Positif": 1, "Netral": 0, "Negatif": -1}

    features = {}
    for asp in aspects:
        mask   = df["aspects"].str.contains(asp, na=False)
        scores = df[mask]["sentiment_rule"].map(sent_score_map).fillna(0)

        # Proporsi positif, netral, negatif sebagai fitur terpisah
        total = mask.sum()
        features[f"{asp}_pos_ratio"] = (
            df[mask]["sentiment_rule"].eq("Positif").sum() / total
            if total > 0 else 0
        )
        features[f"{asp}_neg_ratio"] = (
            df[mask]["sentiment_rule"].eq("Negatif").sum() / total
            if total > 0 else 0
        )
        features[f"{asp}_net_ratio"] = (
            df[mask]["sentiment_rule"].eq("Netral").sum() / total
            if total > 0 else 0
        )
        features[f"{asp}_score"]     = scores.mean() if total > 0 else 0

    feature_df = pd.DataFrame(features, index=[0])
    return feature_df


def run_ridge_regression(
    df: pd.DataFrame,
    aspects: List[str] = ASPECTS,
    output_dir: str = RESULT_DIR
) -> pd.DataFrame:
    """
    Analisis regresi Ridge:
    Menghitung pengaruh sentimen per aspek E-S-QUAL terhadap rating bintang.

    Menggunakan per-review features (bukan agregasi) untuk statistik yang valid.
    """
    logger.info("Menjalankan analisis regresi Ridge ...")

    # Buat feature per ulasan
    rows = []
    for _, row in df.iterrows():
        row_aspects  = str(row.get("aspects", "GENERAL")).split("|")
        sentiment    = row.get("sentiment_rule", "Netral")
        score_mapped = {"Positif": 1, "Netral": 0, "Negatif": -1}.get(sentiment, 0)
        rating       = row.get("rating", 3)

        feat = {"rating": rating}
        for asp in aspects:
            feat[f"{asp}_present"]  = 1 if asp in row_aspects else 0
            feat[f"{asp}_sentiment"] = score_mapped if asp in row_aspects else 0

        rows.append(feat)

    feat_df = pd.DataFrame(rows)

    # Fitur dan target
    feature_cols = [c for c in feat_df.columns if c != "rating"]
    X = feat_df[feature_cols].values
    y = feat_df["rating"].values

    # Standardisasi
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Ridge dengan cross-validation untuk memilih alpha optimal
    alphas = np.logspace(-3, 3, 50)
    ridge_cv = RidgeCV(alphas=alphas, cv=5, scoring="r2")
    ridge_cv.fit(X_scaled, y)

    best_alpha = ridge_cv.alpha_
    logger.info(f"Alpha optimal (Ridge CV): {best_alpha:.4f}")

    # Model final
    ridge = Ridge(alpha=best_alpha)
    ridge.fit(X_scaled, y)

    # Evaluasi
    y_pred = ridge.predict(X_scaled)
    r2     = r2_score(y, y_pred)
    rmse   = np.sqrt(mean_squared_error(y, y_pred))
    cv_r2  = cross_val_score(ridge, X_scaled, y, cv=5, scoring="r2").mean()

    logger.info(f"Regresi Ridge — R²: {r2:.4f} | RMSE: {rmse:.4f} | CV R²: {cv_r2:.4f}")

    # Koefisien
    coef_df = pd.DataFrame({
        "Feature"     : feature_cols,
        "Coefficient" : ridge.coef_,
        "Abs_Coef"    : np.abs(ridge.coef_),
    }).sort_values("Abs_Coef", ascending=False)

    coef_df["Direction"] = coef_df["Coefficient"].apply(
        lambda x: "Positif (↑ Rating)" if x > 0 else "Negatif (↓ Rating)"
    )

    logger.info(f"\nTop 10 Fitur Berpengaruh:\n{coef_df.head(10).to_string(index=False)}")

    # Simpan
    reg_result_path = Path(output_dir) / "regression_analysis.csv"
    coef_df.to_csv(reg_result_path, index=False, encoding="utf-8-sig")

    # ── Visualisasi koefisien ──────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    colors_coef = ["#2ecc71" if c > 0 else "#e74c3c" for c in coef_df["Coefficient"]]
    ax.barh(
        coef_df["Feature"][:15],
        coef_df["Coefficient"][:15],
        color=colors_coef[:15],
        edgecolor="white",
    )
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_xlabel("Koefisien Regresi Ridge", fontsize=11)
    ax.set_title(
        f"Pengaruh Sentimen Aspek E-S-QUAL terhadap Rating\n"
        f"(R² = {r2:.3f}, RMSE = {rmse:.3f}, α = {best_alpha:.4f})",
        fontsize=12, fontweight="bold"
    )
    ax.invert_yaxis()
    plt.tight_layout()

    fig_path = Path(FIGURE_DIR) / "regression_coefficients.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Plot regresi disimpan: {fig_path}")

    return coef_df, {"R2": r2, "RMSE": rmse, "CV_R2": cv_r2, "Alpha": best_alpha}


# ─────────────────────────────────────────────
# 5. PLOT TRAINING CURVES
# ─────────────────────────────────────────────
def plot_training_curves(
    metrics_path: str = METRICS_PATH,
    output_dir: str = FIGURE_DIR
) -> None:
    """Visualisasi kurva training loss dan F1-score per epoch."""
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        log = pd.DataFrame(metrics["training_log"])
        if log.empty:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Kurva Training IndoBERT ABSA — balé by BTN", fontsize=13, fontweight="bold")

        # Loss
        ax1.plot(log["epoch"], log["train_loss"], "o-", color="#003087", label="Train Loss")
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
        ax1.set_title("Training Loss"); ax1.legend(); ax1.grid(alpha=0.3)

        # F1
        if "val_aspect_f1_macro" in log.columns:
            ax2.plot(log["epoch"], log["val_aspect_f1_macro"], "s-", color="#2ecc71", label="Aspect F1")
        if "val_sentiment_f1_avg" in log.columns:
            ax2.plot(log["epoch"], log["val_sentiment_f1_avg"], "^-", color="#e74c3c",  label="Sentiment F1 Avg")
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("F1-Score")
        ax2.set_title("Validation F1-Score"); ax2.legend(); ax2.grid(alpha=0.3)
        ax2.set_ylim(0, 1)

        plt.tight_layout()
        out_path = Path(output_dir) / "training_curves.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Kurva training disimpan: {out_path}")
    except Exception as e:
        logger.warning(f"Tidak bisa plot training curves: {e}")


# ─────────────────────────────────────────────
# 6. RINGKASAN EKSEKUTIF (JSON)
# ─────────────────────────────────────────────
def generate_executive_report(
    df: pd.DataFrame,
    dist_df: pd.DataFrame,
    reg_metrics: Dict,
    output_dir: str = RESULT_DIR
) -> None:
    """Menyusun laporan eksekutif dalam format JSON untuk publikasi."""

    report = {
        "judul_penelitian": "Transforming User Feedback into Strategic Intelligence: "
                            "An ABSA of balé by BTN Superapp using IndoBERT and E-S-QUAL",
        "dataset": {
            "total_ulasan"     : int(len(df)),
            "periode"          : "2025",
            "sumber"           : "Google Play Store",
            "distribusi_rating": df["rating"].value_counts().to_dict(),
        },
        "distribusi_sentimen_per_aspek": {},
        "regresi_ridge": reg_metrics,
        "rekomendasi_strategis": {},
    }

    # Isi distribusi sentimen
    for asp in ASPECTS:
        mask   = df["aspects"].str.contains(asp, na=False)
        subset = df[mask]["sentiment_rule"].value_counts()
        total  = subset.sum()
        report["distribusi_sentimen_per_aspek"][asp] = {
            sent: {"count": int(subset.get(sent, 0)),
                   "pct"  : round(subset.get(sent, 0) / total * 100, 2) if total > 0 else 0}
            for sent in SENTIMENTS
        }

    # Rekomendasi strategis berdasarkan hasil
    for asp in ASPECTS:
        data = report["distribusi_sentimen_per_aspek"][asp]
        neg_pct = data["Negatif"]["pct"]
        pos_pct = data["Positif"]["pct"]

        if neg_pct > 40:
            priority = "🔴 PRIORITAS TINGGI — Perlu perbaikan segera"
        elif neg_pct > 25:
            priority = "🟡 PERHATIAN — Monitoring ketat diperlukan"
        else:
            priority = "🟢 BAIK — Pertahankan dan tingkatkan"

        report["rekomendasi_strategis"][asp] = {
            "status"  : priority,
            "neg_pct" : neg_pct,
            "pos_pct" : pos_pct,
        }

    out_path = Path(output_dir) / "final_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(f"Laporan eksekutif disimpan: {out_path}")

    # Cetak ringkasan ke konsol
    print("\n" + "="*60)
    print("📊 RINGKASAN EKSEKUTIF — balé by BTN ABSA")
    print("="*60)
    for asp, rec in report["rekomendasi_strategis"].items():
        print(f"\n{asp}:")
        print(f"  Status  : {rec['status']}")
        print(f"  Positif : {rec['pos_pct']:.1f}%")
        print(f"  Negatif : {rec['neg_pct']:.1f}%")
    print(f"\n📈 Regresi Ridge R²: {reg_metrics.get('R2', 0):.4f}")
    print("="*60)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import os
    os.makedirs("logs", exist_ok=True)

    # 1. Load data
    logger.info(f"Membaca data dari: {LABELED_PATH}")
    df = pd.read_csv(LABELED_PATH, encoding="utf-8-sig")

    # 2. Distribusi sentimen + visualisasi
    logger.info("Membuat visualisasi distribusi sentimen ...")
    dist_df = plot_sentiment_distribution(df)

    # 3. WordClouds
    logger.info("Membuat word cloud ...")
    plot_wordclouds(df)

    # 4. Training curves (jika ada)
    plot_training_curves()

    # 5. Regresi Ridge
    coef_df, reg_metrics = run_ridge_regression(df)

    # 6. Laporan eksekutif
    generate_executive_report(df, dist_df, reg_metrics)

    print("\n" + "="*60)
    print("✅ STEP 5 SELESAI: Evaluasi & Analisis")
    print(f"   Semua figure disimpan ke : {FIGURE_DIR}")
    print(f"   Laporan final            : {RESULT_DIR}final_report.json")
    print("="*60)
