from __future__ import annotations

from pathlib import Path
from typing import Optional

import chess
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# ============================================================
# CONFIGURATION
# ============================================================

KPK_OUTPUT_ROOT = Path(r"C:\Users\Κωνσταντίνος\Desktop\KPK Dataset")

DATASET_DIR = KPK_OUTPUT_ROOT / "01_dataset"
MODEL_DIR = KPK_OUTPUT_ROOT / "02_model_wdl"
COMPARISON_DIR = KPK_OUTPUT_ROOT / "03_feature_set_comparison"
COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

DATASET_CSV = DATASET_DIR / "kpk_exhaustive_wdl_dataset.csv"
SELECTED_CONFIG_CSV = MODEL_DIR / "kpk_wdl_decision_tree_selected_optimal_depth.csv"

OUTPUT_PREFIX = "kpk_wdl_feature_set_comparison"

CSV_SEPARATOR = ";"
CSV_ENCODING = "utf-8-sig"
DECIMAL_PLACES = 4
FLOAT_FORMAT = f"%.{DECIMAL_PLACES}f"

pd.set_option("display.float_format", lambda value: f"{value:.{DECIMAL_PLACES}f}")

TARGET_COLUMN = "white_is_winning"

SEEDS = [42, 123, 456]
DEPTHS = [7, 9, 12]
CLASS_WEIGHT_OPTIONS = [None, "balanced"]

MIN_SAMPLES_SPLIT = 50
MIN_SAMPLES_LEAF = 25

FALLBACK_DEPTH = 12
FALLBACK_CLASS_WEIGHT: Optional[str] = None


# ============================================================
# FEATURE SETS
# ============================================================

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

FEATURE_SETS = {
    "raw_coordinates": RAW_COORDINATE_FEATURES,
    "geometric_features": GEOMETRIC_FEATURES,
    "full_kpk_strategic_features": FULL_KPK_STRATEGIC_FEATURES,
}

DIAGNOSTIC_COLUMNS = list(
    dict.fromkeys(
        [
            "fen",
            "result_class",
            "wdl_raw",
            TARGET_COLUMN,
            *RAW_COORDINATE_FEATURES,
            *FULL_KPK_STRATEGIC_FEATURES,
        ]
    )
)


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


def save_csv(df: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        path,
        index=index,
        encoding=CSV_ENCODING,
        sep=CSV_SEPARATOR,
        float_format=FLOAT_FORMAT,
        decimal=",",
    )


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


def class_weight_for_filename(class_weight: Optional[str]) -> str:
    normalized = normalize_class_weight_value(class_weight)
    return normalized.lower().replace(" ", "_")


def safe_int(value, default: int = 0) -> int:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return default
    return int(numeric)


# ============================================================
# RAW COORDINATE EXTRACTION FROM FEN
# ============================================================

def extract_raw_coordinates_from_fen(fen: str) -> dict[str, int]:
    board = chess.Board(fen)

    wk = board.king(chess.WHITE)
    bk = board.king(chess.BLACK)
    white_pawns = list(board.pieces(chess.PAWN, chess.WHITE))

    if wk is None or bk is None or len(white_pawns) != 1:
        raise ValueError(f"Invalid KPK FEN for raw coordinate extraction: {fen}")

    wp = white_pawns[0]

    return {
        "wk_file": chess.square_file(wk),
        "wk_rank": chess.square_rank(wk),
        "wp_file": chess.square_file(wp),
        "wp_rank": chess.square_rank(wp),
        "bk_file": chess.square_file(bk),
        "bk_rank": chess.square_rank(bk),
    }


def ensure_raw_coordinate_columns(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()

    missing_raw_columns = [
        col for col in ["wk_file", "wk_rank", "wp_file", "wp_rank", "bk_file", "bk_rank"]
        if col not in output.columns
    ]

    if not missing_raw_columns:
        return output

    if "fen" not in output.columns:
        raise ValueError(
            "Raw coordinate features are missing and cannot be reconstructed because 'fen' is missing."
        )

    raw_rows = output["fen"].apply(extract_raw_coordinates_from_fen)
    raw_df = pd.DataFrame(raw_rows.tolist(), index=output.index)

    for col in raw_df.columns:
        output[col] = raw_df[col]

    return output


# ============================================================
# SELECTED MAIN CONFIGURATION
# ============================================================

def load_selected_main_config() -> tuple[int, Optional[str], pd.DataFrame]:
    if not SELECTED_CONFIG_CSV.exists():
        print(
            f"\nSelected config file not found: {SELECTED_CONFIG_CSV}\n"
            f"Using fallback config: depth={FALLBACK_DEPTH}, class_weight={FALLBACK_CLASS_WEIGHT}"
        )

        fallback_class_weight_str = normalize_class_weight_value(FALLBACK_CLASS_WEIGHT)

        config_df = pd.DataFrame(
            [
                {
                    "main_depth": FALLBACK_DEPTH,
                    "main_class_weight": fallback_class_weight_str,
                    "source": (
                        f"FALLBACK: depth={FALLBACK_DEPTH}, "
                        f"class_weight={fallback_class_weight_str}"
                    ),
                }
            ]
        )

        return FALLBACK_DEPTH, FALLBACK_CLASS_WEIGHT, config_df

    selected_df = read_csv_auto(SELECTED_CONFIG_CSV)

    if selected_df.empty:
        raise ValueError(f"Selected config file is empty: {SELECTED_CONFIG_CSV}")

    row = selected_df.iloc[0]

    if "optimal_depth" not in selected_df.columns:
        raise ValueError(
            f"Missing 'optimal_depth' in selected config file: {SELECTED_CONFIG_CSV}"
        )

    selected_depth = int(row["optimal_depth"])

    if "optimal_class_weight" in selected_df.columns:
        selected_class_weight_str = normalize_class_weight_value(row["optimal_class_weight"])
    elif "class_weight" in selected_df.columns:
        selected_class_weight_str = normalize_class_weight_value(row["class_weight"])
    else:
        selected_class_weight_str = normalize_class_weight_value(FALLBACK_CLASS_WEIGHT)

    selected_class_weight = parse_class_weight_for_sklearn(selected_class_weight_str)

    config_df = pd.DataFrame(
        [
            {
                "main_depth": selected_depth,
                "main_class_weight": selected_class_weight_str,
                "source": (
                    f"AUTO from KPKDecTree: depth={selected_depth}, "
                    f"class_weight={selected_class_weight_str}"
                ),
            }
        ]
    )

    return selected_depth, selected_class_weight, config_df

# ============================================================
# VALIDATION
# ============================================================

def validate_feature_sets(df: pd.DataFrame) -> None:
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    for feature_set_name, features in FEATURE_SETS.items():
        missing = [col for col in features if col not in df.columns]
        if missing:
            raise ValueError(
                f"Missing features for {feature_set_name}: {missing}\n"
                f"Available columns: {df.columns.tolist()}"
            )


# ============================================================
# ERROR CATEGORIZATION
# ============================================================

def classify_error_row(row: pd.Series) -> tuple[str, str]:
    true_label = safe_int(row["true"])
    pred_label = safe_int(row["pred"])
    flags: list[str] = []

    is_rook_pawn = safe_int(row.get("is_rook_pawn", 0))
    black_inside_square = safe_int(row.get("black_king_inside_square_of_pawn", 0))
    black_can_block = safe_int(row.get("black_king_can_block_promotion", 0))
    white_wins_key_race = safe_int(row.get("white_wins_key_square_race", 0))
    black_ahead = safe_int(row.get("black_king_ahead_of_pawn", 0))
    direct_opposition = safe_int(row.get("kings_have_direct_opposition", 0))
    distant_opposition = safe_int(row.get("kings_have_distant_opposition", 0))
    white_strong_support = safe_int(row.get("white_king_strongly_supports_pawn", 0))
    steps = safe_int(row.get("steps_to_promotion", -1), default=-1)
    pawn_distance_diff = safe_int(row.get("pawn_distance_diff", 0))
    wk_pawn_distance = safe_int(row.get("white_king_pawn_distance", 99), default=99)

    if true_label == 0 and pred_label == 1:
        if is_rook_pawn:
            flags.append("rook_pawn_draw_missed")
        if black_inside_square:
            flags.append("square_rule_draw_missed")
        if black_can_block:
            flags.append("promotion_blockade_draw_missed")
        if black_ahead:
            flags.append("black_king_ahead_draw_missed")
        if direct_opposition or distant_opposition:
            flags.append("opposition_tempo_draw_missed")
        if not flags:
            flags.append("other_false_positive_win_prediction")

    elif true_label == 1 and pred_label == 0:
        if white_wins_key_race:
            flags.append("key_square_win_missed")
        if not black_inside_square and not black_can_block:
            flags.append("promotion_race_win_missed")
        if not is_rook_pawn and steps <= 3:
            flags.append("advanced_non_rook_pawn_win_missed")
        if pawn_distance_diff > 0 and wk_pawn_distance <= 2:
            flags.append("white_king_support_win_missed")
        if direct_opposition or distant_opposition:
            flags.append("opposition_tempo_win_missed")
        if white_strong_support:
            flags.append("strong_king_support_win_missed")
        if not flags:
            flags.append("other_false_negative_draw_prediction")

    else:
        flags.append("not_an_error")

    return flags[0], " | ".join(flags)


def categorize_errors(errors_df: pd.DataFrame) -> pd.DataFrame:
    categorized = errors_df.copy()

    if categorized.empty:
        categorized["error_type"] = pd.Series(dtype="object")
        categorized["primary_error_category"] = pd.Series(dtype="object")
        categorized["all_error_flags"] = pd.Series(dtype="object")
        return categorized

    pairs = categorized.apply(classify_error_row, axis=1)

    categorized["primary_error_category"] = [pair[0] for pair in pairs]
    categorized["all_error_flags"] = [pair[1] for pair in pairs]
    categorized["error_type"] = categorized.apply(
        lambda row: "false_positive_predicted_win"
        if safe_int(row["true"]) == 0 and safe_int(row["pred"]) == 1
        else "false_negative_predicted_draw",
        axis=1,
    )

    return categorized


# ============================================================
# MODEL TRAINING
# ============================================================

def make_classifier(
    max_depth: int,
    class_weight: Optional[str],
    seed: int,
) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(
        criterion="gini",
        max_depth=max_depth,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        class_weight=class_weight,
        random_state=seed,
    )


def add_diagnostics(
    results: pd.DataFrame,
    df: pd.DataFrame,
    indices: pd.Index,
) -> pd.DataFrame:
    enriched = results.copy()

    for col in DIAGNOSTIC_COLUMNS:
        if col in df.columns and col not in enriched.columns:
            enriched[col] = df.loc[indices, col].values

    return enriched


def train_single_tree(
    df: pd.DataFrame,
    *,
    feature_set_name: str,
    feature_columns: list[str],
    max_depth: int,
    class_weight: Optional[str],
    seed: int,
) -> dict:
    X = df[feature_columns]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=seed,
        stratify=y,
    )

    clf = make_classifier(
        max_depth=max_depth,
        class_weight=class_weight,
        seed=seed,
    )

    clf.fit(X_train, y_train)

    y_train_pred = clf.predict(X_train)
    y_pred = clf.predict(X_test)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    results = X_test.copy()
    results["true"] = y_test.values
    results["pred"] = y_pred
    results = add_diagnostics(results, df, X_test.index)

    errors = results[results["true"] != results["pred"]].copy()
    categorized_errors = categorize_errors(errors)

    importances = pd.DataFrame(
        {
            "feature_set": feature_set_name,
            "feature": feature_columns,
            "importance": clf.feature_importances_,
            "seed": seed,
            "max_depth": max_depth,
            "class_weight": normalize_class_weight_value(class_weight),
        }
    ).sort_values("importance", ascending=False)

    return {
        "feature_set": feature_set_name,
        "seed": seed,
        "max_depth": max_depth,
        "class_weight": class_weight,
        "train_accuracy": accuracy_score(y_train, y_train_pred),
        "test_accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro"),
        "weighted_f1": f1_score(y_test, y_pred, average="weighted"),
        "actual_depth": clf.get_depth(),
        "number_of_leaves": clf.get_n_leaves(),
        "true_draw_pred_draw": int(cm[0, 0]),
        "true_draw_pred_win": int(cm[0, 1]),
        "true_win_pred_draw": int(cm[1, 0]),
        "true_win_pred_win": int(cm[1, 1]),
        "total_errors": len(errors),
        "importances": importances,
        "categorized_errors": categorized_errors,
    }


# ============================================================
# COMPARISON / SUMMARY
# ============================================================

def run_comparison(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_rows: list[dict] = []
    importances: list[pd.DataFrame] = []
    error_category_rows: list[dict] = []

    for feature_set_name, feature_columns in FEATURE_SETS.items():
        print_section(f"FEATURE SET: {feature_set_name}")

        for seed in SEEDS:
            for depth in DEPTHS:
                for class_weight in CLASS_WEIGHT_OPTIONS:
                    result = train_single_tree(
                        df,
                        feature_set_name=feature_set_name,
                        feature_columns=feature_columns,
                        max_depth=depth,
                        class_weight=class_weight,
                        seed=seed,
                    )

                    class_weight_label = normalize_class_weight_value(class_weight)

                    raw_rows.append(
                        {
                            "feature_set": feature_set_name,
                            "seed": seed,
                            "max_depth": depth,
                            "class_weight": class_weight_label,
                            "train_accuracy": result["train_accuracy"],
                            "test_accuracy": result["test_accuracy"],
                            "accuracy_gap": result["train_accuracy"] - result["test_accuracy"],
                            "balanced_accuracy": result["balanced_accuracy"],
                            "macro_f1": result["macro_f1"],
                            "weighted_f1": result["weighted_f1"],
                            "actual_depth": result["actual_depth"],
                            "number_of_leaves": result["number_of_leaves"],
                            "true_draw_pred_draw": result["true_draw_pred_draw"],
                            "true_draw_pred_win": result["true_draw_pred_win"],
                            "true_win_pred_draw": result["true_win_pred_draw"],
                            "true_win_pred_win": result["true_win_pred_win"],
                            "false_positive_count": result["true_draw_pred_win"],
                            "false_negative_count": result["true_win_pred_draw"],
                            "total_errors": result["total_errors"],
                            "top_feature": result["importances"].iloc[0]["feature"],
                        }
                    )

                    importances.append(result["importances"])

                    if not result["categorized_errors"].empty:
                        counts = (
                            result["categorized_errors"]["primary_error_category"]
                            .value_counts()
                            .reset_index()
                        )
                        counts.columns = ["primary_error_category", "count"]

                        for _, row in counts.iterrows():
                            error_category_rows.append(
                                {
                                    "feature_set": feature_set_name,
                                    "seed": seed,
                                    "max_depth": depth,
                                    "class_weight": class_weight_label,
                                    "primary_error_category": row["primary_error_category"],
                                    "count": int(row["count"]),
                                }
                            )

                    print(
                        f"seed={seed} | depth={depth} | class_weight={class_weight_label} | "
                        f"acc={result['test_accuracy']:.4f} | "
                        f"bal_acc={result['balanced_accuracy']:.4f} | "
                        f"macro_f1={result['macro_f1']:.4f} | "
                        f"errors={result['total_errors']} | leaves={result['number_of_leaves']}"
                    )

    raw_results = pd.DataFrame(raw_rows)
    raw_importances = pd.concat(importances, ignore_index=True) if importances else pd.DataFrame()
    raw_error_categories = pd.DataFrame(error_category_rows)

    return raw_results, raw_importances, raw_error_categories


def summarize_results(raw_results: pd.DataFrame) -> pd.DataFrame:
    return (
        raw_results.groupby(["feature_set", "max_depth", "class_weight"])
        .agg(
            test_accuracy_mean=("test_accuracy", "mean"),
            test_accuracy_std=("test_accuracy", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            weighted_f1_mean=("weighted_f1", "mean"),
            weighted_f1_std=("weighted_f1", "std"),
            total_errors_mean=("total_errors", "mean"),
            total_errors_std=("total_errors", "std"),
            false_positive_count_mean=("false_positive_count", "mean"),
            false_negative_count_mean=("false_negative_count", "mean"),
            number_of_leaves_mean=("number_of_leaves", "mean"),
            number_of_leaves_std=("number_of_leaves", "std"),
        )
        .reset_index()
    )


def create_main_table(
    summary: pd.DataFrame,
    *,
    main_depth: int,
    main_class_weight: Optional[str],
) -> pd.DataFrame:
    main_class_weight_label = normalize_class_weight_value(main_class_weight)

    main = summary[
        (summary["max_depth"] == main_depth)
        & (summary["class_weight"].apply(normalize_class_weight_value) == main_class_weight_label)
    ].copy()

    if main.empty:
        raise ValueError(
            f"No rows found for selected main comparison config: "
            f"depth={main_depth}, class_weight={main_class_weight_label}"
        )

    order = {
        "raw_coordinates": 1,
        "geometric_features": 2,
        "full_kpk_strategic_features": 3,
    }

    main["feature_set_order"] = main["feature_set"].map(order)
    main = main.sort_values("feature_set_order").drop(columns=["feature_set_order"])

    raw_row = main[main["feature_set"] == "raw_coordinates"]
    geometric_row = main[main["feature_set"] == "geometric_features"]

    if not raw_row.empty:
        raw_errors = float(raw_row.iloc[0]["total_errors_mean"])
        raw_balanced_accuracy = float(raw_row.iloc[0]["balanced_accuracy_mean"])
        raw_macro_f1 = float(raw_row.iloc[0]["macro_f1_mean"])

        main["error_reduction_vs_raw"] = raw_errors - main["total_errors_mean"]
        main["balanced_accuracy_gain_vs_raw"] = main["balanced_accuracy_mean"] - raw_balanced_accuracy
        main["macro_f1_gain_vs_raw"] = main["macro_f1_mean"] - raw_macro_f1

    if not geometric_row.empty:
        geometric_errors = float(geometric_row.iloc[0]["total_errors_mean"])
        geometric_balanced_accuracy = float(geometric_row.iloc[0]["balanced_accuracy_mean"])
        geometric_macro_f1 = float(geometric_row.iloc[0]["macro_f1_mean"])

        main["error_reduction_vs_geometric"] = geometric_errors - main["total_errors_mean"]
        main["balanced_accuracy_gain_vs_geometric"] = main["balanced_accuracy_mean"] - geometric_balanced_accuracy
        main["macro_f1_gain_vs_geometric"] = main["macro_f1_mean"] - geometric_macro_f1

    return main


def summarize_importances(importances: pd.DataFrame) -> pd.DataFrame:
    if importances.empty:
        return pd.DataFrame()

    return (
        importances.groupby(["feature_set", "feature"])
        .agg(
            importance_mean=("importance", "mean"),
            importance_std=("importance", "std"),
            importance_min=("importance", "min"),
            importance_max=("importance", "max"),
        )
        .reset_index()
        .sort_values(["feature_set", "importance_mean"], ascending=[True, False])
    )


def summarize_error_categories(error_categories: pd.DataFrame) -> pd.DataFrame:
    if error_categories.empty:
        return pd.DataFrame()

    return (
        error_categories.groupby(
            [
                "feature_set",
                "max_depth",
                "class_weight",
                "primary_error_category",
            ]
        )
        .agg(
            count_mean=("count", "mean"),
            count_std=("count", "std"),
            count_min=("count", "min"),
            count_max=("count", "max"),
        )
        .reset_index()
        .sort_values(
            ["feature_set", "max_depth", "class_weight", "count_mean"],
            ascending=[True, True, True, False],
        )
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print_section("KPK WDL FEATURE SET COMPARISON")
    print(f"Dataset: {DATASET_CSV}")
    print(f"Selected config: {SELECTED_CONFIG_CSV}")
    print(f"Output folder: {COMPARISON_DIR}")

    main_depth, main_class_weight, config_used = load_selected_main_config()
    main_class_weight_label = normalize_class_weight_value(main_class_weight)

    print("\nMain comparison configuration:")
    print(f"- max_depth: {main_depth}")
    print(f"- class_weight: {main_class_weight_label}")

    df = read_csv_auto(DATASET_CSV)
    df = ensure_raw_coordinate_columns(df)
    validate_feature_sets(df)

    raw_results, raw_importances, raw_error_categories = run_comparison(df)

    stability_summary = summarize_results(raw_results)
    main_comparison = create_main_table(
        stability_summary,
        main_depth=main_depth,
        main_class_weight=main_class_weight,
    )
    mean_importances = summarize_importances(raw_importances)
    error_category_summary = summarize_error_categories(raw_error_categories)

    main_filename = (
        f"{OUTPUT_PREFIX}_main_depth{main_depth}_"
        f"{class_weight_for_filename(main_class_weight)}_comparison.csv"
    )

    print_section("MAIN COMPARISON - SELECTED CONFIG")
    print(main_comparison.to_string(index=False))

    save_csv(config_used, COMPARISON_DIR / f"{OUTPUT_PREFIX}_main_config_used.csv")
    save_csv(raw_results, COMPARISON_DIR / f"{OUTPUT_PREFIX}_raw_results.csv")
    save_csv(stability_summary, COMPARISON_DIR / f"{OUTPUT_PREFIX}_stability_summary.csv")
    save_csv(main_comparison, COMPARISON_DIR / main_filename)
    save_csv(raw_importances, COMPARISON_DIR / f"{OUTPUT_PREFIX}_raw_importances.csv")
    save_csv(mean_importances, COMPARISON_DIR / f"{OUTPUT_PREFIX}_mean_importances.csv")
    save_csv(raw_error_categories, COMPARISON_DIR / f"{OUTPUT_PREFIX}_error_categories_raw.csv")
    save_csv(error_category_summary, COMPARISON_DIR / f"{OUTPUT_PREFIX}_error_category_summary.csv")

    print_section("FILES SAVED")
    for path in sorted(COMPARISON_DIR.glob(f"{OUTPUT_PREFIX}_*")):
        print(path)


if __name__ == "__main__":
    main()
