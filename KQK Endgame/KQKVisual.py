from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree


# ============================================================
# CONFIGURATION
# ============================================================

KQK_OUTPUT_ROOT = Path(r"C:\Users\Κωνσταντίνος\Desktop\KQK Dataset")

DATASET_DIR = KQK_OUTPUT_ROOT / "01_dataset"
MODEL_DIR = KQK_OUTPUT_ROOT / "02_model_dtz"
COMPARISON_DIR = KQK_OUTPUT_ROOT / "03_feature_set_comparison"
FIGURES_DIR = KQK_OUTPUT_ROOT / "04_figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DATASET_CSV = DATASET_DIR / "kqk_exhaustive_dtz_dataset.csv"

MODEL_PREFIX = "kqk_dtz_decision_tree"
COMPARISON_PREFIX = "kqk_feature_set_comparison"
CSV_SEPARATOR = ";"

TARGET_COLUMN = "dtz_bucket"

PHASE_ORDER = [
    "long_conversion",
    "medium_conversion",
    "short_conversion",
    "immediate_mate",
]

PHASE_SHORT_NAMES = {
    "long_conversion": "Long",
    "medium_conversion": "Medium",
    "short_conversion": "Short",
    "immediate_mate": "Immediate",
}

MAIN_SEED = 42

# Selected automatically by KQKDecTree.py
DECTREE_SELECTED_CONFIG_CSV = MODEL_DIR / "kqk_dtz_decision_tree_selected_optimal_depth.csv"

# Used only if selected config CSV is missing or malformed
FALLBACK_MAIN_DEPTH = 9
FALLBACK_MAIN_CLASS_WEIGHT: Optional[str] = "balanced"

MIN_SAMPLES_SPLIT = 50
MIN_SAMPLES_LEAF = 25

FEATURE_COLUMNS = [
    "bk_legal_moves_if_black_to_move",
    "bk_distance_to_corner",
    "bk_distance_to_edge",
    "wk_distance_to_edge",
    "wk_wq_distance",
    "wk_bk_chebyshev_distance",
    "wk_bk_manhattan_distance",
    "wq_bk_chebyshev_distance",
    "queen_box_area",
    "queen_controls_bk_zone_count",
    "stalemate_risk_level",
]

FEATURE_SET_LABELS = {
    "raw_coordinates": "Raw coordinates",
    "geometric_features": "Geometric",
    "full_kqk_strategic_features": "Full KQK Strategic",
}

plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["font.family"] = "DejaVu Sans"


# ============================================================
# UTILITIES
# ============================================================

def print_section(title: str, width: int = 90) -> None:
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def convert_numeric_like_columns(df: pd.DataFrame) -> pd.DataFrame:
    converted_df = df.copy()

    for col in converted_df.columns:
        if converted_df[col].dtype != "object":
            continue

        series = converted_df[col].astype(str).str.strip()
        numeric_candidate = series.str.replace(",", ".", regex=False)
        converted = pd.to_numeric(numeric_candidate, errors="coerce")

        original_notna = converted_df[col].notna()
        converted_notna = converted.notna()

        if converted_notna[original_notna].all():
            converted_df[col] = converted

    return converted_df


def read_csv_auto(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(
        path,
        sep=CSV_SEPARATOR,
        decimal=",",
        encoding="utf-8-sig",
    )

    if len(df.columns) == 1:
        df = pd.read_csv(path, encoding="utf-8-sig")

    return convert_numeric_like_columns(df)


def save_figure(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved figure: {output_path}")


def display_feature_set_name(name: str) -> str:
    return FEATURE_SET_LABELS.get(name, name)


def normalize_class_weight_value(value) -> str:
    if pd.isna(value):
        return "None"

    value_str = str(value).strip()

    if value_str == "" or value_str.lower() in {"none", "nan", "null"}:
        return "None"

    return value_str


def parse_class_weight_for_sklearn(class_weight_str: str) -> Optional[str]:
    if class_weight_str == "None":
        return None

    return class_weight_str


def load_main_config_from_dectree() -> tuple[int, Optional[str], str]:
    """
    Read the optimal (depth, class_weight) selected by KQKDecTree.py.

    Falls back only if the selected-config CSV is missing, unreadable, or malformed.
    """

    if not DECTREE_SELECTED_CONFIG_CSV.exists():
        print(
            f"\n[WARNING] DecTree selection CSV not found:\n"
            f"  {DECTREE_SELECTED_CONFIG_CSV}\n"
            f"Falling back to depth={FALLBACK_MAIN_DEPTH}, "
            f"class_weight={FALLBACK_MAIN_CLASS_WEIGHT}."
        )

        return (
            FALLBACK_MAIN_DEPTH,
            FALLBACK_MAIN_CLASS_WEIGHT,
            "FALLBACK (DecTree CSV missing)",
        )

    try:
        selection_df = read_csv_auto(DECTREE_SELECTED_CONFIG_CSV)
    except Exception as exc:
        print(
            f"\n[WARNING] Could not read DecTree selection CSV: {exc}\n"
            f"Falling back to depth={FALLBACK_MAIN_DEPTH}, "
            f"class_weight={FALLBACK_MAIN_CLASS_WEIGHT}."
        )

        return (
            FALLBACK_MAIN_DEPTH,
            FALLBACK_MAIN_CLASS_WEIGHT,
            "FALLBACK (read error)",
        )

    required_columns = {"optimal_depth", "optimal_class_weight"}
    missing_columns = required_columns - set(selection_df.columns)

    if missing_columns or selection_df.empty:
        print(
            f"\n[WARNING] DecTree selection CSV malformed. Missing columns: {missing_columns}\n"
            f"Falling back to depth={FALLBACK_MAIN_DEPTH}, "
            f"class_weight={FALLBACK_MAIN_CLASS_WEIGHT}."
        )

        return (
            FALLBACK_MAIN_DEPTH,
            FALLBACK_MAIN_CLASS_WEIGHT,
            "FALLBACK (malformed CSV)",
        )

    first_row = selection_df.iloc[0]

    main_depth = int(first_row["optimal_depth"])
    main_class_weight_str = normalize_class_weight_value(first_row["optimal_class_weight"])
    main_class_weight = parse_class_weight_for_sklearn(main_class_weight_str)

    return (
        main_depth,
        main_class_weight,
        f"AUTO from KQKDecTree: depth={main_depth}, class_weight={main_class_weight}",
    )


def validate_dataset_columns(df: pd.DataFrame) -> None:
    required_columns = [TARGET_COLUMN, *FEATURE_COLUMNS]
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(
            "Missing required columns for KQKVisual.py:\n"
            f"{missing}\n\n"
            "This usually means that KQKGenerator.py has not been rerun "
            "after the latest feature changes."
        )


# ============================================================
# TREE RETRAINING
# ============================================================

def train_main_tree_for_visualization(
    df: pd.DataFrame,
    main_depth: int,
    main_class_weight: Optional[str],
) -> DecisionTreeClassifier:
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=MAIN_SEED,
        stratify=y,
    )

    clf = DecisionTreeClassifier(
        criterion="gini",
        max_depth=main_depth,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        class_weight=main_class_weight,
        random_state=MAIN_SEED,
    )

    clf.fit(X_train, y_train)

    return clf


# ============================================================
# PLOTS
# ============================================================

def plot_dataset_distribution(df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    bucket_counts = df[TARGET_COLUMN].value_counts().reindex(PHASE_ORDER).fillna(0)
    short_names = [PHASE_SHORT_NAMES[p] for p in PHASE_ORDER]

    axes[0].bar(short_names, bucket_counts.values)
    axes[0].set_ylabel("Count", fontsize=12)
    axes[0].set_title("DTZ Bucket Distribution", fontsize=13)
    axes[0].grid(axis="y", alpha=0.3)

    total = bucket_counts.sum()

    for i, value in enumerate(bucket_counts.values):
        percentage = 100 * value / total if total > 0 else 0
        axes[0].text(
            i,
            value + total * 0.01,
            f"{int(value):,}\n({percentage:.1f}%)",
            ha="center",
            fontsize=9,
        )

    if "mate_moves" in df.columns:
        max_mate_moves = int(df["mate_moves"].max())

        axes[1].hist(
            df["mate_moves"],
            bins=range(0, max_mate_moves + 2),
            edgecolor="black",
        )

        axes[1].set_xlabel("Approximate mate moves", fontsize=12)
        axes[1].set_ylabel("Count", fontsize=12)
        axes[1].set_title("Mate-Move Distribution", fontsize=13)
        axes[1].grid(axis="y", alpha=0.3)

        for boundary in [1.5, 3.5, 6.5]:
            axes[1].axvline(
                boundary,
                linestyle="--",
                alpha=0.6,
                linewidth=1,
            )

    elif "dtz_abs" in df.columns:
        max_dtz_abs = int(df["dtz_abs"].max())

        axes[1].hist(
            df["dtz_abs"],
            bins=range(0, max_dtz_abs + 2),
            edgecolor="black",
        )

        axes[1].set_xlabel("DTZ absolute", fontsize=12)
        axes[1].set_ylabel("Count", fontsize=12)
        axes[1].set_title("Raw DTZ Distribution", fontsize=13)
        axes[1].grid(axis="y", alpha=0.3)

    save_figure(output_path)


def plot_decision_tree_top_levels(
    clf: DecisionTreeClassifier,
    output_path: Path,
    max_depth: int = 3,
) -> None:
    fig, ax = plt.subplots(figsize=(30, 15))

    class_names = [PHASE_SHORT_NAMES.get(c, c) for c in clf.classes_]

    plot_tree(
        clf,
        feature_names=FEATURE_COLUMNS,
        class_names=class_names,
        filled=True,
        rounded=True,
        max_depth=max_depth,
        fontsize=8,
        ax=ax,
    )

    ax.set_title(
        f"KQK Decision Tree — Top {max_depth} Levels",
        fontsize=16,
        pad=20,
    )

    save_figure(output_path)


def plot_feature_importances(
    importances: pd.DataFrame,
    output_path: Path,
    top_n: int = 12,
) -> None:
    if importances.empty:
        print("Importances empty, skipping plot.")
        return

    top = importances.head(top_n).iloc[::-1].copy()

    fig, ax = plt.subplots(figsize=(12, 8))

    bars = ax.barh(top["feature"], top["importance"])

    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(f"Top {top_n} Feature Importances — KQK", fontsize=14)
    ax.grid(axis="x", alpha=0.3)

    max_imp = top["importance"].max()

    for bar, value in zip(bars, top["importance"]):
        ax.text(
            value + max_imp * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            fontsize=9,
        )

    save_figure(output_path)


def plot_confusion_matrix(confusion_df: pd.DataFrame, output_path: Path) -> None:
    if "true_label" in confusion_df.columns:
        cm = confusion_df.drop(columns=["true_label"]).to_numpy()
    else:
        cm = confusion_df.to_numpy()

    cm = cm.astype(int)

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm,
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0,
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    short_names = [PHASE_SHORT_NAMES[p] for p in PHASE_ORDER]

    im0 = axes[0].imshow(cm)
    axes[0].set_title("Confusion Matrix — Counts", fontsize=13)
    axes[0].set_xlabel("Predicted", fontsize=12)
    axes[0].set_ylabel("True", fontsize=12)
    axes[0].set_xticks(range(len(short_names)))
    axes[0].set_yticks(range(len(short_names)))
    axes[0].set_xticklabels(short_names, rotation=15)
    axes[0].set_yticklabels(short_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[0].text(
                j,
                i,
                f"{cm[i, j]:,}",
                ha="center",
                va="center",
                fontsize=10,
            )

    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(cm_norm, vmin=0, vmax=1)
    axes[1].set_title("Confusion Matrix — Row Normalized", fontsize=13)
    axes[1].set_xlabel("Predicted", fontsize=12)
    axes[1].set_ylabel("True", fontsize=12)
    axes[1].set_xticks(range(len(short_names)))
    axes[1].set_yticks(range(len(short_names)))
    axes[1].set_xticklabels(short_names, rotation=15)
    axes[1].set_yticklabels(short_names)

    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            axes[1].text(
                j,
                i,
                f"{cm_norm[i, j]:.1%}",
                ha="center",
                va="center",
                fontsize=10,
            )

    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    save_figure(output_path)


def plot_phase_confusion(
    phase_confusion: pd.DataFrame,
    output_path: Path,
) -> None:
    if phase_confusion.empty:
        return

    df = phase_confusion.head(15).copy()

    df["label"] = (
        df["true"].map(PHASE_SHORT_NAMES)
        + " → "
        + df["pred"].map(PHASE_SHORT_NAMES)
    )

    fig, ax = plt.subplots(figsize=(11, 7))

    ax.barh(df["label"][::-1], df["count"][::-1])
    ax.set_xlabel("Count", fontsize=12)
    ax.set_title("Top 7 Phase Confusions (True → Predicted)", fontsize=14)
    ax.grid(axis="x", alpha=0.3)

    save_figure(output_path)


def plot_error_categories(
    error_counts: pd.DataFrame,
    output_path: Path,
) -> None:
    if error_counts.empty:
        return

    sorted_df = error_counts.sort_values("count", ascending=True).copy()

    fig, ax = plt.subplots(figsize=(12, 9))

    ax.barh(sorted_df["primary_error_category"], sorted_df["count"])
    ax.set_xlabel("Count", fontsize=12)
    ax.set_title("Error Categories — KQK", fontsize=14)
    ax.grid(axis="x", alpha=0.3)

    max_c = sorted_df["count"].max()

    for i, value in enumerate(sorted_df["count"]):
        ax.text(
            value + max_c * 0.01,
            i,
            f"{value}",
            va="center",
            fontsize=9,
        )

    save_figure(output_path)


def plot_error_severity(
    severity_counts: pd.DataFrame,
    output_path: Path,
) -> None:
    if severity_counts.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.bar(
        severity_counts["absolute_phase_distance"].astype(str),
        severity_counts["count"],
    )

    ax.set_xlabel("Absolute Phase Distance", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Error Severity by Phase Distance", fontsize=14)
    ax.grid(axis="y", alpha=0.3)

    max_count = severity_counts["count"].max()

    for i, value in enumerate(severity_counts["count"]):
        ax.text(
            i,
            value + max_count * 0.01,
            f"{value}",
            ha="center",
            fontsize=10,
        )

    save_figure(output_path)


def plot_errors_by_value(
    errors_by_value: pd.DataFrame,
    value_column: str,
    output_path: Path,
) -> None:
    if errors_by_value.empty or value_column not in errors_by_value.columns:
        return

    pivot = (
        errors_by_value
        .pivot_table(
            index=value_column,
            columns="error_type",
            values="count",
            aggfunc="sum",
            fill_value=0,
        )
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    bottom = np.zeros(len(pivot))

    for column in pivot.columns:
        values = pivot[column].to_numpy()

        ax.bar(
            pivot.index.astype(str),
            values,
            bottom=bottom,
            label=column,
        )

        bottom += values

    ax.set_xlabel(value_column, fontsize=12)
    ax.set_ylabel("Error count", fontsize=12)
    ax.set_title(f"Errors by {value_column}", fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    if len(pivot.index) > 20:
        step = max(1, len(pivot.index) // 15)

        for i, label in enumerate(ax.get_xticklabels()):
            label.set_visible(i % step == 0)

    save_figure(output_path)


def plot_feature_set_comparison(
    main_comparison: pd.DataFrame,
    output_path: Path,
) -> None:
    if main_comparison.empty:
        return

    plot_df = main_comparison.copy()

    feature_set_order = [
        "raw_coordinates",
        "geometric_features",
        "full_kqk_strategic_features",
    ]

    plot_df["feature_set_order"] = plot_df["feature_set"].apply(
        lambda name: feature_set_order.index(name)
        if name in feature_set_order
        else len(feature_set_order)
    )

    plot_df = (
        plot_df
        .sort_values("feature_set_order")
        .drop(columns=["feature_set_order"])
    )

    plot_df["display_name"] = plot_df["feature_set"].map(display_feature_set_name)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    x_pos = np.arange(len(plot_df))

    bal_acc = plot_df["balanced_accuracy_mean"].to_numpy()
    bal_acc_std = plot_df["balanced_accuracy_std"].to_numpy()

    axes[0].bar(
        x_pos,
        bal_acc,
        yerr=bal_acc_std,
        capsize=5,
        alpha=0.85,
    )

    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(
        plot_df["display_name"],
        rotation=10,
        ha="right",
    )

    axes[0].set_ylabel("Balanced Accuracy", fontsize=12)
    axes[0].set_title(
        "Balanced Accuracy by Feature Set\n(mean ± std)",
        fontsize=13,
    )
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].set_ylim([max(0.0, min(bal_acc) - 0.05), 1.0])

    for i, (mean, std) in enumerate(zip(bal_acc, bal_acc_std)):
        axes[0].text(
            i,
            mean + std + 0.005,
            f"{mean:.3f}",
            ha="center",
            fontsize=10,
        )

    errors = plot_df["total_errors_mean"].to_numpy()
    errors_std = plot_df["total_errors_std"].to_numpy()

    axes[1].bar(
        x_pos,
        errors,
        yerr=errors_std,
        capsize=5,
        alpha=0.85,
    )

    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(
        plot_df["display_name"],
        rotation=10,
        ha="right",
    )

    axes[1].set_ylabel("Total Errors", fontsize=12)
    axes[1].set_title(
        "Total Errors by Feature Set\n(mean ± std)",
        fontsize=13,
    )
    axes[1].grid(axis="y", alpha=0.3)

    max_errors = max(errors) if len(errors) else 0

    for i, (mean, std) in enumerate(zip(errors, errors_std)):
        axes[1].text(
            i,
            mean + std + max_errors * 0.02,
            f"{mean:.0f}",
            ha="center",
            fontsize=10,
        )

    save_figure(output_path)


def plot_stability_across_depths(
    raw_results: pd.DataFrame,
    output_path: Path,
    main_class_weight_str: str,
) -> None:
    """
    Plot balanced accuracy versus tree depth, using the same class_weight
    selected by KQKDecTree.py for the main pipeline.
    """

    if raw_results.empty:
        return

    raw = raw_results.copy()
    raw["class_weight_normalized"] = raw["class_weight"].apply(
        normalize_class_weight_value
    )

    subset = raw[
        raw["class_weight_normalized"] == main_class_weight_str
    ].copy()

    if subset.empty:
        print(
            f"No rows found for class_weight={main_class_weight_str}. "
            f"Stability plot skipped."
        )
        return

    feature_set_order = [
        "raw_coordinates",
        "geometric_features",
        "full_kqk_strategic_features",
    ]

    fig, ax = plt.subplots(figsize=(12, 7))

    for feature_set in feature_set_order:
        feature_subset = subset[subset["feature_set"] == feature_set]

        if feature_subset.empty:
            continue

        grouped = (
            feature_subset
            .groupby("max_depth")["balanced_accuracy"]
            .agg(["mean", "std"])
            .reset_index()
        )

        ax.errorbar(
            grouped["max_depth"],
            grouped["mean"],
            yerr=grouped["std"].fillna(0),
            marker="o",
            capsize=5,
            label=display_feature_set_name(feature_set),
            linewidth=2,
            markersize=7,
        )

    ax.set_xlabel("Max Depth", fontsize=12)
    ax.set_ylabel("Balanced Accuracy", fontsize=12)
    ax.set_title(
        "Balanced Accuracy vs Tree Depth\nper Feature Set",
        fontsize=13,
    )
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)

    save_figure(output_path)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print_section("KQK VISUALIZATION MODULE")
    print(f"Dataset folder:    {DATASET_DIR}")
    print(f"Model folder:      {MODEL_DIR}")
    print(f"Comparison folder: {COMPARISON_DIR}")
    print(f"Figures folder:    {FIGURES_DIR}")

    print_section("RESOLVING MAIN CONFIG FROM KQKDecTree")

    main_depth, main_class_weight, source_description = load_main_config_from_dectree()
    main_class_weight_str = normalize_class_weight_value(main_class_weight)

    print(source_description)
    print(f"main_depth = {main_depth}")
    print(f"main_class_weight = {main_class_weight}")

    print_section("DATASET DISTRIBUTION")

    df = read_csv_auto(DATASET_CSV)
    validate_dataset_columns(df)

    plot_dataset_distribution(
        df=df,
        output_path=FIGURES_DIR / "kqk_dataset_distribution.png",
    )

    print_section("DECISION TREE TOP LEVELS")

    clf = train_main_tree_for_visualization(
        df=df,
        main_depth=main_depth,
        main_class_weight=main_class_weight,
    )

    plot_decision_tree_top_levels(
        clf=clf,
        output_path=FIGURES_DIR / "kqk_decision_tree_top3.png",
        max_depth=3,
    )

    print_section("FEATURE IMPORTANCES")

    imp_path = MODEL_DIR / f"{MODEL_PREFIX}_main_model_importances.csv"

    if imp_path.exists():
        plot_feature_importances(
            importances=read_csv_auto(imp_path),
            output_path=FIGURES_DIR / "kqk_feature_importances.png",
        )
    else:
        print(f"Missing: {imp_path}")

    print_section("CONFUSION MATRIX")

    cm_path = MODEL_DIR / f"{MODEL_PREFIX}_confusion_matrix.csv"

    if cm_path.exists():
        plot_confusion_matrix(
            confusion_df=read_csv_auto(cm_path),
            output_path=FIGURES_DIR / "kqk_confusion_matrix.png",
        )
    else:
        print(f"Missing: {cm_path}")

    print_section("PHASE CONFUSIONS")

    pc_path = MODEL_DIR / f"{MODEL_PREFIX}_phase_confusion_counts.csv"

    if pc_path.exists():
        plot_phase_confusion(
            phase_confusion=read_csv_auto(pc_path),
            output_path=FIGURES_DIR / "kqk_phase_confusion_counts.png",
        )
    else:
        print(f"Missing: {pc_path}")

    print_section("ERROR CATEGORIES")

    ec_path = MODEL_DIR / f"{MODEL_PREFIX}_error_category_counts.csv"

    if ec_path.exists():
        plot_error_categories(
            error_counts=read_csv_auto(ec_path),
            output_path=FIGURES_DIR / "kqk_error_categories.png",
        )
    else:
        print(f"Missing: {ec_path}")

    print_section("ERROR SEVERITY")

    es_path = MODEL_DIR / f"{MODEL_PREFIX}_error_severity_counts.csv"

    if es_path.exists():
        plot_error_severity(
            severity_counts=read_csv_auto(es_path),
            output_path=FIGURES_DIR / "kqk_error_severity.png",
        )
    else:
        print(f"Missing: {es_path}")

    print_section("ERRORS BY MATE MOVES")

    emm_path = MODEL_DIR / f"{MODEL_PREFIX}_errors_by_mate_moves.csv"

    if emm_path.exists():
        plot_errors_by_value(
            errors_by_value=read_csv_auto(emm_path),
            value_column="mate_moves",
            output_path=FIGURES_DIR / "kqk_errors_by_mate_moves.png",
        )
    else:
        print(f"Missing: {emm_path}")

    print_section("ERRORS BY DTZ ABS")

    edtz_path = MODEL_DIR / f"{MODEL_PREFIX}_errors_by_dtz_abs.csv"

    if edtz_path.exists():
        plot_errors_by_value(
            errors_by_value=read_csv_auto(edtz_path),
            value_column="dtz_abs",
            output_path=FIGURES_DIR / "kqk_errors_by_dtz_abs.png",
        )
    else:
        print(f"Missing: {edtz_path}")

    print_section("FEATURE SET COMPARISON")

    fc_filename = (
        f"{COMPARISON_PREFIX}_main_depth{main_depth}_"
        f"{main_class_weight_str.lower()}_comparison.csv"
    )

    fc_path = COMPARISON_DIR / fc_filename

    if fc_path.exists():
        plot_feature_set_comparison(
            main_comparison=read_csv_auto(fc_path),
            output_path=FIGURES_DIR / "kqk_feature_set_comparison.png",
        )
    else:
        print(f"Missing: {fc_path}")
        print("Hint: re-run KQKComp.py after KQKDecTree.py.")

    print_section("STABILITY ACROSS DEPTHS")

    sd_path = COMPARISON_DIR / f"{COMPARISON_PREFIX}_raw_results.csv"

    if sd_path.exists():
        plot_stability_across_depths(
            raw_results=read_csv_auto(sd_path),
            output_path=FIGURES_DIR / "kqk_stability_across_depths.png",
            main_class_weight_str=main_class_weight_str,
        )
    else:
        print(f"Missing: {sd_path}")

    print_section("FIGURES CREATED")

    for path in sorted(FIGURES_DIR.glob("*.png")):
        print(path)


if __name__ == "__main__":
    main()