from __future__ import annotations

from pathlib import Path
from typing import Optional

import chess
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree


# ============================================================
# CONFIGURATION
# ============================================================

KPK_OUTPUT_ROOT = Path(r"C:\Users\Κωνσταντίνος\Desktop\KPK Dataset")

DATASET_DIR = KPK_OUTPUT_ROOT / "01_dataset"
MODEL_DIR = KPK_OUTPUT_ROOT / "02_model_wdl"
COMPARISON_DIR = KPK_OUTPUT_ROOT / "03_feature_set_comparison"
FIGURES_DIR = KPK_OUTPUT_ROOT / "04_figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DATASET_CSV = DATASET_DIR / "kpk_exhaustive_wdl_dataset.csv"

MODEL_PREFIX = "kpk_wdl_decision_tree"
COMPARISON_PREFIX = "kpk_wdl_feature_set_comparison"

SELECTED_CONFIG_CSV = MODEL_DIR / f"{MODEL_PREFIX}_selected_optimal_depth.csv"

CSV_SEPARATOR = ";"
CSV_ENCODING = "utf-8-sig"
DECIMAL_PLACES = 4
FLOAT_FORMAT = f"%.{DECIMAL_PLACES}f"

TARGET_COLUMN = "white_is_winning"
CLASS_NAMES = ["Draw", "Win"]

MAIN_SEED = 42
FALLBACK_DEPTH = 12
FALLBACK_CLASS_WEIGHT: Optional[str] = None

MIN_SAMPLES_SPLIT = 50
MIN_SAMPLES_LEAF = 25

RAW_COORDINATE_FEATURES = [
    "side_to_move",
    "wk_file",
    "wk_rank",
    "wp_file",
    "wp_rank",
    "bk_file",
    "bk_rank",
]

GEOMETRIC_FEATURES = [
    "side_to_move",
    "steps_to_promotion",
    "is_rook_pawn",
    "black_king_ahead_of_pawn",
    "white_king_pawn_distance",
    "black_king_pawn_distance",
    "pawn_distance_diff",
    "kings_distance",
]

FULL_KPK_STRATEGIC_FEATURES = [
    *GEOMETRIC_FEATURES,
    "black_king_inside_square_of_pawn",
    "black_king_can_block_promotion",
    "white_wins_key_square_race",
    "kings_have_direct_opposition",
    "kings_have_distant_opposition",
    "white_king_strongly_supports_pawn",
]

FEATURE_COLUMNS = FULL_KPK_STRATEGIC_FEATURES

FEATURE_SET_LABELS = {
    "raw_coordinates": "Raw Coordinates",
    "geometric_features": "Geometric",
    "full_kpk_strategic_features": "Full KPK Strategic",
}

plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["font.family"] = "DejaVu Sans"

pd.set_option("display.float_format", lambda value: f"{value:.{DECIMAL_PLACES}f}")


# ============================================================
# UTILITY FUNCTIONS
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
    """Read semicolon-separated CSVs, with fallback for older comma-separated files."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(
        path,
        sep=CSV_SEPARATOR,
        decimal=",",
        encoding=CSV_ENCODING,
    )

    if len(df.columns) == 1:
        df = pd.read_csv(path, encoding=CSV_ENCODING)

    return convert_numeric_like_columns(df)


def save_figure(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved figure: {output_path}")


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


def format_class_weight_for_filename(class_weight: Optional[str]) -> str:
    if class_weight is None:
        return "None"
    return str(class_weight)


def display_feature_set_name(name: str) -> str:
    return FEATURE_SET_LABELS.get(name, name)


# ============================================================
# SELECTED MAIN CONFIG
# ============================================================

def load_selected_main_config() -> tuple[int, Optional[str], pd.DataFrame]:
    """
    Load the automatically selected main-model configuration from KPKDecTree.

    If the selected config file is missing, fall back to the historical default.
    """
    if not SELECTED_CONFIG_CSV.exists():
        config_df = pd.DataFrame([
            {
                "main_depth": FALLBACK_DEPTH,
                "main_class_weight": str(FALLBACK_CLASS_WEIGHT),
                "source": "FALLBACK: selected config file not found",
            }
        ])
        return FALLBACK_DEPTH, FALLBACK_CLASS_WEIGHT, config_df

    selected = read_csv_auto(SELECTED_CONFIG_CSV)

    if "optimal_depth" in selected.columns:
        main_depth = int(selected.loc[0, "optimal_depth"])
    elif "main_depth" in selected.columns:
        main_depth = int(selected.loc[0, "main_depth"])
    else:
        raise ValueError(
            f"Could not find optimal_depth/main_depth in {SELECTED_CONFIG_CSV}. "
            f"Available columns: {selected.columns.tolist()}"
        )

    if "optimal_class_weight" in selected.columns:
        class_weight_str = normalize_class_weight_value(selected.loc[0, "optimal_class_weight"])
    elif "main_class_weight" in selected.columns:
        class_weight_str = normalize_class_weight_value(selected.loc[0, "main_class_weight"])
    elif "class_weight" in selected.columns:
        class_weight_str = normalize_class_weight_value(selected.loc[0, "class_weight"])
    else:
        class_weight_str = "None"

    main_class_weight = parse_class_weight_for_sklearn(class_weight_str)

    config_df = pd.DataFrame([
        {
            "main_depth": main_depth,
            "main_class_weight": class_weight_str,
            "source": f"AUTO from KPKDecTree: depth={main_depth}, class_weight={class_weight_str}",
        }
    ])

    return main_depth, main_class_weight, config_df


# ============================================================
# RAW COORDINATE SUPPORT
# ============================================================

def add_raw_coordinate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add wk/wp/bk file-rank coordinate columns from FEN if they are missing."""
    required = {"wk_file", "wk_rank", "wp_file", "wp_rank", "bk_file", "bk_rank"}

    if required.issubset(df.columns):
        return df

    if "fen" not in df.columns:
        raise ValueError("Raw coordinate columns are missing and FEN column is unavailable.")

    enriched = df.copy()

    wk_files: list[int] = []
    wk_ranks: list[int] = []
    wp_files: list[int] = []
    wp_ranks: list[int] = []
    bk_files: list[int] = []
    bk_ranks: list[int] = []

    for fen in enriched["fen"]:
        board = chess.Board(str(fen))

        wk = board.king(chess.WHITE)
        bk = board.king(chess.BLACK)
        white_pawns = list(board.pieces(chess.PAWN, chess.WHITE))

        if wk is None or bk is None or len(white_pawns) != 1:
            raise ValueError(f"Invalid KPK FEN while extracting coordinates: {fen}")

        wp = white_pawns[0]

        wk_files.append(chess.square_file(wk))
        wk_ranks.append(chess.square_rank(wk))
        wp_files.append(chess.square_file(wp))
        wp_ranks.append(chess.square_rank(wp))
        bk_files.append(chess.square_file(bk))
        bk_ranks.append(chess.square_rank(bk))

    enriched["wk_file"] = wk_files
    enriched["wk_rank"] = wk_ranks
    enriched["wp_file"] = wp_files
    enriched["wp_rank"] = wp_ranks
    enriched["bk_file"] = bk_files
    enriched["bk_rank"] = bk_ranks

    return enriched


# ============================================================
# MODEL RE-TRAINING FOR TREE VISUALIZATION
# ============================================================

def train_main_tree_for_visualization(
    df: pd.DataFrame,
    *,
    main_depth: int,
    main_class_weight: Optional[str],
) -> DecisionTreeClassifier:
    """
    Re-train the selected main decision tree only for visualization.

    This does not change the experimental results. With the same seed, selected
    depth, class_weight and feature columns, it reproduces the main model setup.
    """
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

def plot_dataset_distribution(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Plot:
    1. target distribution: Draw vs Win
    2. steps to promotion distribution by target class
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    target_counts = df[TARGET_COLUMN].value_counts().sort_index()
    labels = ["Draw", "Win"]

    axes[0].bar(labels, target_counts.values)
    axes[0].set_ylabel("Count", fontsize=12)
    axes[0].set_title("Target Distribution: Draw vs Win", fontsize=13)

    total = target_counts.sum()

    for i, value in enumerate(target_counts.values):
        percentage = 100 * value / total
        axes[0].text(
            i,
            value + total * 0.01,
            f"{value:,}\n({percentage:.1f}%)",
            ha="center",
            fontsize=10,
        )

    if "steps_to_promotion" in df.columns:
        for label_value, label_name in [(0, "Draw"), (1, "Win")]:
            subset = df[df[TARGET_COLUMN] == label_value]
            axes[1].hist(
                subset["steps_to_promotion"],
                bins=np.arange(1, 9) - 0.5,
                alpha=0.65,
                label=label_name,
                edgecolor="black",
            )

        axes[1].set_xlabel("Steps to Promotion", fontsize=12)
        axes[1].set_ylabel("Count", fontsize=12)
        axes[1].set_title("Steps to Promotion by Outcome", fontsize=13)
        axes[1].set_xticks(range(1, 8))
        axes[1].legend()

    save_figure(output_path)


def plot_decision_tree_top_levels(
    clf: DecisionTreeClassifier,
    output_path: Path,
    max_depth: int = 3,
) -> None:
    """Plot only the first levels of the decision tree."""
    _fig, ax = plt.subplots(figsize=(28, 14))

    plot_tree(
        clf,
        feature_names=FEATURE_COLUMNS,
        class_names=CLASS_NAMES,
        filled=True,
        rounded=True,
        max_depth=max_depth,
        fontsize=9,
        ax=ax,
    )

    ax.set_title(
        f"KPK WDL Decision Tree — Top {max_depth} Levels",
        fontsize=16,
        pad=20,
    )

    save_figure(output_path)


def plot_feature_importances(
    importances: pd.DataFrame,
    output_path: Path,
    top_n: int = 15,
) -> None:
    """Horizontal bar chart of top feature importances."""
    if importances.empty:
        print("Feature importances file is empty. Skipping plot.")
        return

    top = importances.head(top_n).iloc[::-1].copy()

    _fig, ax = plt.subplots(figsize=(11, 8))

    bars = ax.barh(
        top["feature"],
        top["importance"],
    )

    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(f"Top {top_n} Feature Importances — KPK WDL", fontsize=14)
    ax.grid(axis="x", alpha=0.3)

    max_importance = top["importance"].max()

    for bar, value in zip(bars, top["importance"]):
        ax.text(
            value + max_importance * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            fontsize=9,
        )

    save_figure(output_path)


def plot_confusion_matrix(
    confusion_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot confusion matrix as raw counts and row-normalized percentages."""
    if "true_label" in confusion_df.columns:
        cm = confusion_df.drop(columns=["true_label"]).to_numpy()
    else:
        cm = confusion_df.to_numpy()

    cm = cm.astype(int)

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_normalized = np.divide(
        cm,
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0,
    )

    _fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    im0 = axes[0].imshow(cm)
    axes[0].set_title("Confusion Matrix — Counts", fontsize=13)
    axes[0].set_xlabel("Predicted", fontsize=12)
    axes[0].set_ylabel("True", fontsize=12)
    axes[0].set_xticks([0, 1])
    axes[0].set_yticks([0, 1])
    axes[0].set_xticklabels(CLASS_NAMES)
    axes[0].set_yticklabels(CLASS_NAMES)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[0].text(
                j,
                i,
                f"{cm[i, j]:,}",
                ha="center",
                va="center",
                fontsize=12,
            )

    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(cm_normalized, vmin=0, vmax=1)
    axes[1].set_title("Confusion Matrix — Row Normalized", fontsize=13)
    axes[1].set_xlabel("Predicted", fontsize=12)
    axes[1].set_ylabel("True", fontsize=12)
    axes[1].set_xticks([0, 1])
    axes[1].set_yticks([0, 1])
    axes[1].set_xticklabels(CLASS_NAMES)
    axes[1].set_yticklabels(CLASS_NAMES)

    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            axes[1].text(
                j,
                i,
                f"{cm_normalized[i, j]:.2%}",
                ha="center",
                va="center",
                fontsize=12,
            )

    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    save_figure(output_path)


def plot_error_categories(
    error_counts: pd.DataFrame,
    output_path: Path,
) -> None:
    """Bar chart of categorized errors by chess motif."""
    if error_counts.empty:
        print("Error category counts file is empty. Skipping plot.")
        return

    required_columns = {"primary_error_category", "count"}

    if not required_columns.issubset(error_counts.columns):
        print(f"Missing columns in error_counts: {required_columns}. Skipping plot.")
        return

    sorted_df = error_counts.sort_values("count", ascending=True).copy()

    _fig, ax = plt.subplots(figsize=(12, 8))

    ax.barh(
        sorted_df["primary_error_category"],
        sorted_df["count"],
    )

    ax.set_xlabel("Count", fontsize=12)
    ax.set_title("Error Categories by Chess Motif — KPK WDL", fontsize=14)
    ax.grid(axis="x", alpha=0.3)

    max_count = sorted_df["count"].max()

    for i, value in enumerate(sorted_df["count"]):
        ax.text(
            value + max_count * 0.01,
            i,
            f"{value}",
            va="center",
            fontsize=9,
        )

    save_figure(output_path)


def plot_feature_set_comparison(
    main_comparison: pd.DataFrame,
    output_path: Path,
) -> None:
    """Bar chart comparing balanced accuracy and total errors across feature sets."""
    if main_comparison.empty:
        print("Main comparison file is empty. Skipping plot.")
        return

    required_columns = {
        "feature_set",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
        "total_errors_mean",
        "total_errors_std",
    }

    if not required_columns.issubset(main_comparison.columns):
        print(f"Missing columns in main_comparison: {required_columns}. Skipping plot.")
        return

    plot_df = main_comparison.copy()
    plot_df["display_name"] = plot_df["feature_set"].map(display_feature_set_name)

    order = {
        "raw_coordinates": 1,
        "geometric_features": 2,
        "full_kpk_strategic_features": 3,
    }
    plot_df["feature_set_order"] = plot_df["feature_set"].map(order)
    plot_df = plot_df.sort_values("feature_set_order")

    _fig, axes = plt.subplots(1, 2, figsize=(16, 6))

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
    axes[0].set_xticklabels(plot_df["display_name"], rotation=10, ha="right")
    axes[0].set_ylabel("Balanced Accuracy", fontsize=12)
    axes[0].set_title("Balanced Accuracy by Feature Set\nmean ± std across seeds", fontsize=13)
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
    axes[1].set_xticklabels(plot_df["display_name"], rotation=10, ha="right")
    axes[1].set_ylabel("Total Errors", fontsize=12)
    axes[1].set_title("Total Errors by Feature Set\nmean ± std across seeds", fontsize=13)
    axes[1].grid(axis="y", alpha=0.3)

    for i, (mean, std) in enumerate(zip(errors, errors_std)):
        axes[1].text(
            i,
            mean + std + max(errors) * 0.02,
            f"{mean:.0f}",
            ha="center",
            fontsize=10,
        )

    save_figure(output_path)


def plot_stability_across_depths(
    raw_results: pd.DataFrame,
    output_path: Path,
) -> None:
    """Line plot of balanced accuracy vs tree depth per feature set."""
    if raw_results.empty:
        print("Raw comparison results file is empty. Skipping plot.")
        return

    required_columns = {
        "feature_set",
        "max_depth",
        "class_weight",
        "balanced_accuracy",
    }

    if not required_columns.issubset(raw_results.columns):
        print(f"Missing columns in raw_results: {required_columns}. Skipping plot.")
        return

    raw_results = raw_results.copy()
    raw_results["class_weight_normalized"] = raw_results["class_weight"].apply(
        normalize_class_weight_value
    )

    none_results = raw_results[
        raw_results["class_weight_normalized"] == "None"
    ].copy()

    if none_results.empty:
        print("No class_weight == None rows found. Skipping stability plot.")
        return

    _fig, ax = plt.subplots(figsize=(12, 7))

    order = [
        "raw_coordinates",
        "geometric_features",
        "full_kpk_strategic_features",
    ]

    for feature_set in order:
        subset = none_results[none_results["feature_set"] == feature_set]

        if subset.empty:
            continue

        grouped = (
            subset
            .groupby("max_depth")["balanced_accuracy"]
            .agg(["mean", "std"])
            .reset_index()
        )

        ax.errorbar(
            grouped["max_depth"],
            grouped["mean"],
            yerr=grouped["std"],
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
    print_section("KPK WDL VISUALIZATION MODULE")

    print(f"Dataset folder: {DATASET_DIR}")
    print(f"Model folder: {MODEL_DIR}")
    print(f"Comparison folder: {COMPARISON_DIR}")
    print(f"Figures folder: {FIGURES_DIR}")

    main_depth, main_class_weight, config_df = load_selected_main_config()
    main_class_weight_for_filename = format_class_weight_for_filename(main_class_weight)

    print("\nSelected main configuration:")
    print(config_df.to_string(index=False))

    # --------------------------------------------------------
    # Dataset distribution
    # --------------------------------------------------------
    print_section("DATASET DISTRIBUTION")

    df = read_csv_auto(DATASET_CSV)
    df = add_raw_coordinate_columns(df)

    plot_dataset_distribution(
        df=df,
        output_path=FIGURES_DIR / "kpk_wdl_dataset_distribution.png",
    )

    # --------------------------------------------------------
    # Decision tree top levels
    # --------------------------------------------------------
    print_section("DECISION TREE TOP LEVELS")

    clf = train_main_tree_for_visualization(
        df=df,
        main_depth=main_depth,
        main_class_weight=main_class_weight,
    )

    plot_decision_tree_top_levels(
        clf=clf,
        output_path=FIGURES_DIR / "kpk_wdl_decision_tree_top3.png",
        max_depth=3,
    )

    # --------------------------------------------------------
    # Feature importances
    # --------------------------------------------------------
    print_section("FEATURE IMPORTANCES")

    importances_path = MODEL_DIR / f"{MODEL_PREFIX}_main_model_importances.csv"

    if importances_path.exists():
        importances = read_csv_auto(importances_path)
        plot_feature_importances(
            importances=importances,
            output_path=FIGURES_DIR / "kpk_wdl_feature_importances.png",
            top_n=15,
        )
    else:
        print(f"Missing file: {importances_path}")

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------
    print_section("CONFUSION MATRIX")

    confusion_path = MODEL_DIR / f"{MODEL_PREFIX}_confusion_matrix.csv"

    if confusion_path.exists():
        confusion_df = read_csv_auto(confusion_path)
        plot_confusion_matrix(
            confusion_df=confusion_df,
            output_path=FIGURES_DIR / "kpk_wdl_confusion_matrix.png",
        )
    else:
        print(f"Missing file: {confusion_path}")

    # --------------------------------------------------------
    # Error categories
    # --------------------------------------------------------
    print_section("ERROR CATEGORIES")

    error_counts_path = MODEL_DIR / f"{MODEL_PREFIX}_error_category_counts.csv"

    if error_counts_path.exists():
        error_counts = read_csv_auto(error_counts_path)
        plot_error_categories(
            error_counts=error_counts,
            output_path=FIGURES_DIR / "kpk_wdl_error_categories.png",
        )
    else:
        print(f"Missing file: {error_counts_path}")

    # --------------------------------------------------------
    # Feature set comparison
    # --------------------------------------------------------
    print_section("FEATURE SET COMPARISON")

    main_comparison_path = (
        COMPARISON_DIR
        / f"{COMPARISON_PREFIX}_main_depth{main_depth}_{main_class_weight_for_filename}_comparison.csv"
    )

    if main_comparison_path.exists():
        main_comparison = read_csv_auto(main_comparison_path)
        plot_feature_set_comparison(
            main_comparison=main_comparison,
            output_path=FIGURES_DIR / "kpk_wdl_feature_set_comparison.png",
        )
    else:
        print(f"Missing file: {main_comparison_path}")

    # --------------------------------------------------------
    # Stability across depths
    # --------------------------------------------------------
    print_section("STABILITY ACROSS DEPTHS")

    raw_comparison_path = COMPARISON_DIR / f"{COMPARISON_PREFIX}_raw_results.csv"

    if raw_comparison_path.exists():
        raw_results = read_csv_auto(raw_comparison_path)
        plot_stability_across_depths(
            raw_results=raw_results,
            output_path=FIGURES_DIR / "kpk_wdl_stability_across_depths.png",
        )
    else:
        print(f"Missing file: {raw_comparison_path}")

    print_section("VISUALIZATION FILES CREATED")

    for path in sorted(FIGURES_DIR.glob("kpk_wdl_*.png")):
        print(path)


if __name__ == "__main__":
    main()
