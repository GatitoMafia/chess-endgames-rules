from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# ============================================================
# CONFIGURATION
# ============================================================

KQK_OUTPUT_ROOT = Path(r"C:\Users\Κωνσταντίνος\Desktop\KQK Dataset")
DATASET_DIR = KQK_OUTPUT_ROOT / "01_dataset"
MODEL_DIR = KQK_OUTPUT_ROOT / "02_model_dtz"
COMPARISON_DIR = KQK_OUTPUT_ROOT / "03_feature_set_comparison"
COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

DATASET_CSV = DATASET_DIR / "kqk_exhaustive_dtz_dataset.csv"

# Path to the auto-selection CSV written by the upgraded KQKDecTree.
DECTREE_SELECTED_CONFIG_CSV = MODEL_DIR / "kqk_dtz_decision_tree_selected_optimal_depth.csv"

OUTPUT_PREFIX = "kqk_feature_set_comparison"
CSV_SEPARATOR = ";"
FLOAT_DECIMALS = 4
FLOAT_FORMAT = f"%.{FLOAT_DECIMALS}f"

pd.options.display.float_format = f"{{:.{FLOAT_DECIMALS}f}}".format

TARGET_COLUMN = "dtz_bucket"

SEEDS = [42, 123, 456]
DEPTHS = [7, 9, 12]
CLASS_WEIGHT_OPTIONS = [None, "balanced"]

# Fallback main config used only if the DecTree selection CSV is missing.
FALLBACK_MAIN_DEPTH = 9
FALLBACK_MAIN_CLASS_WEIGHT: Optional[str] = "balanced"

PHASE_ORDER = [
    "long_conversion",
    "medium_conversion",
    "short_conversion",
    "immediate_mate",
]


# ============================================================
# FEATURE SETS
# ============================================================

RAW_COORDINATES_FEATURES = [
    "wk_file", "wk_rank",
    "wq_file", "wq_rank",
    "bk_file", "bk_rank",
]

GEOMETRIC_FEATURES = [
    "bk_distance_to_corner",
    "bk_distance_to_edge",
    "wk_distance_to_edge",
    "wk_wq_distance",
    "wk_bk_chebyshev_distance",
    "wk_bk_manhattan_distance",
    "wq_bk_chebyshev_distance",
    "queen_box_area",
]

FULL_KQK_STRATEGIC_FEATURES = [
    *GEOMETRIC_FEATURES,
    "bk_legal_moves_if_black_to_move",
    "queen_controls_bk_zone_count",
    "stalemate_risk_level",
]

FEATURE_SETS = {
    "raw_coordinates": RAW_COORDINATES_FEATURES,
    "geometric_features": GEOMETRIC_FEATURES,
    "full_kqk_strategic_features": FULL_KQK_STRATEGIC_FEATURES,
}

DIAGNOSTIC_COLUMNS = list(dict.fromkeys([
    "fen", "wdl", "dtz", "dtz_abs", "mate_moves", "dtz_bucket",
    *FULL_KQK_STRATEGIC_FEATURES,
    *RAW_COORDINATES_FEATURES,
]))


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
    df = pd.read_csv(path, sep=CSV_SEPARATOR, decimal=",", encoding="utf-8-sig")
    if len(df.columns) == 1:
        df = pd.read_csv(path, encoding="utf-8-sig")
    return convert_numeric_like_columns(df)


def save_csv(df: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index, encoding="utf-8-sig", sep=CSV_SEPARATOR, float_format=FLOAT_FORMAT, decimal=",")


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
    Read the optimal (depth, class_weight) that KQKDecTree selected automatically
    via adjusted_score and use it as the main comparison config. Falls back to
    hardcoded defaults if the selection CSV is not available.
    """
    if not DECTREE_SELECTED_CONFIG_CSV.exists():
        print(f"\n[WARNING] DecTree selection CSV not found at:\n  {DECTREE_SELECTED_CONFIG_CSV}"
              f"\nFalling back to defaults: depth={FALLBACK_MAIN_DEPTH}, class_weight={FALLBACK_MAIN_CLASS_WEIGHT}.")
        return FALLBACK_MAIN_DEPTH, FALLBACK_MAIN_CLASS_WEIGHT, "FALLBACK (DecTree CSV missing)"
    try:
        selection_df = read_csv_auto(DECTREE_SELECTED_CONFIG_CSV)
    except Exception as exc:
        print(f"\n[WARNING] Could not read DecTree selection CSV: {exc}\nFalling back to defaults.")
        return FALLBACK_MAIN_DEPTH, FALLBACK_MAIN_CLASS_WEIGHT, "FALLBACK (read error)"
    required = {"optimal_depth", "optimal_class_weight"}
    if required - set(selection_df.columns) or selection_df.empty:
        print("\n[WARNING] DecTree selection CSV malformed. Falling back to defaults.")
        return FALLBACK_MAIN_DEPTH, FALLBACK_MAIN_CLASS_WEIGHT, "FALLBACK (malformed CSV)"
    first_row = selection_df.iloc[0]
    main_depth = int(first_row["optimal_depth"])
    main_class_weight_str = normalize_class_weight_value(first_row["optimal_class_weight"])
    main_class_weight = parse_class_weight_for_sklearn(main_class_weight_str)
    return main_depth, main_class_weight, f"AUTO from KQKDecTree: depth={main_depth}, class_weight={main_class_weight}"


def validate_feature_sets(df: pd.DataFrame) -> None:
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")
    for name, features in FEATURE_SETS.items():
        missing = [col for col in features if col not in df.columns]
        if missing:
            raise ValueError(f"Missing features for {name}: {missing}\nAvailable columns: {df.columns.tolist()}")


def phase_to_rank(phase: str) -> int:
    return {phase: idx for idx, phase in enumerate(PHASE_ORDER)}.get(str(phase), -1)


# ============================================================
# ERROR CATEGORIZATION
# ============================================================

def classify_error_row(row: pd.Series) -> tuple[str, str]:
    true_phase = str(row["true"])
    pred_phase = str(row["pred"])
    true_rank = phase_to_rank(true_phase)
    pred_rank = phase_to_rank(pred_phase)
    flags: list[str] = []

    if true_rank == -1 or pred_rank == -1:
        return "unknown_phase_confusion", "unknown_phase_confusion"

    phase_distance = pred_rank - true_rank
    if phase_distance > 0:
        error_type = "predicted_too_close_to_mate"
    elif phase_distance < 0:
        error_type = "predicted_too_far_from_mate"
    else:
        error_type = "same_phase_no_error"
    flags.append(error_type)

    if abs(phase_distance) >= 2:
        flags.append("large_phase_confusion")

    pair = {true_phase, pred_phase}
    if pair == {"long_conversion", "medium_conversion"}:
        flags.append("long_medium_boundary_confusion")
    elif pair == {"medium_conversion", "short_conversion"}:
        flags.append("medium_short_boundary_confusion")
    elif pair == {"short_conversion", "immediate_mate"}:
        flags.append("short_immediate_boundary_confusion")
    elif pair == {"long_conversion", "short_conversion"}:
        flags.append("long_short_large_confusion")
    elif pair == {"medium_conversion", "immediate_mate"}:
        flags.append("medium_immediate_large_confusion")
    elif pair == {"long_conversion", "immediate_mate"}:
        flags.append("long_immediate_extreme_confusion")

    bk_legal_moves = int(row.get("bk_legal_moves_if_black_to_move", -1))
    bk_distance_to_edge = int(row.get("bk_distance_to_edge", -1))
    wk_bk_chebyshev = int(row.get("wk_bk_chebyshev_distance", -1))
    queen_box_area = int(row.get("queen_box_area", -1))
    queen_controls = int(row.get("queen_controls_bk_zone_count", -1))
    stalemate_risk = int(row.get("stalemate_risk_level", 0))

    if phase_distance > 0:
        if bk_legal_moves >= 4:
            flags.append("black_king_freedom_overestimated_as_restriction")
        if bk_distance_to_edge >= 2:
            flags.append("edge_distance_overestimated_as_restriction")
        if wk_bk_chebyshev >= 4:
            flags.append("white_king_distance_underestimated")
        if queen_box_area >= 25:
            flags.append("queen_box_size_underestimated")
        if 0 <= queen_controls <= 2:
            flags.append("queen_control_overestimated")

    if phase_distance < 0:
        if 0 <= bk_legal_moves <= 2:
            flags.append("black_king_restriction_underestimated")
        if bk_distance_to_edge <= 0:
            flags.append("edge_restriction_underestimated")
        if 0 <= wk_bk_chebyshev <= 2:
            flags.append("white_king_support_underestimated")
        if 0 <= queen_box_area <= 12:
            flags.append("queen_box_restriction_underestimated")
        if queen_controls >= 4:
            flags.append("queen_control_underestimated")


    priority_order = [
        "long_immediate_extreme_confusion",
        "medium_immediate_large_confusion",
        "long_short_large_confusion",
        "short_immediate_boundary_confusion",
        "medium_short_boundary_confusion",
        "long_medium_boundary_confusion",
        "black_king_freedom_overestimated_as_restriction",
        "edge_distance_overestimated_as_restriction",
        "queen_box_size_underestimated",
        "queen_control_overestimated",
        "white_king_distance_underestimated",
        "black_king_restriction_underestimated",
        "edge_restriction_underestimated",
        "queen_box_restriction_underestimated",
        "queen_control_underestimated",
        "white_king_support_underestimated",
        "large_phase_confusion",
        "predicted_too_close_to_mate",
        "predicted_too_far_from_mate",
    ]
    primary_category = next((candidate for candidate in priority_order if candidate in flags), error_type)
    return primary_category, " | ".join(dict.fromkeys(flags))


def categorize_errors(errors_df: pd.DataFrame) -> pd.DataFrame:
    categorized = errors_df.copy()
    if categorized.empty:
        for col in ["error_type", "primary_error_category", "all_error_flags"]:
            categorized[col] = pd.Series(dtype="object")
        for col in ["true_rank", "pred_rank", "phase_distance", "absolute_phase_distance"]:
            categorized[col] = pd.Series(dtype="float")
        return categorized

    pairs = categorized.apply(classify_error_row, axis=1)
    categorized["primary_error_category"] = [pair[0] for pair in pairs]
    categorized["all_error_flags"] = [pair[1] for pair in pairs]
    categorized["true_rank"] = categorized["true"].apply(phase_to_rank)
    categorized["pred_rank"] = categorized["pred"].apply(phase_to_rank)
    categorized["phase_distance"] = categorized["pred_rank"] - categorized["true_rank"]
    categorized["absolute_phase_distance"] = categorized["phase_distance"].abs()
    categorized["error_type"] = categorized["phase_distance"].apply(
        lambda diff: "predicted_too_close_to_mate" if diff > 0 else "predicted_too_far_from_mate" if diff < 0 else "same_phase_no_error"
    )
    return categorized


# ============================================================
# MODEL TRAINING
# ============================================================

def make_classifier(max_depth: int, class_weight: Optional[str], seed: int) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(
        criterion="gini",
        max_depth=max_depth,
        min_samples_split=50,
        min_samples_leaf=25,
        class_weight=class_weight,
        random_state=seed,
    )


def train_single_tree(df: pd.DataFrame, *, feature_set_name: str, feature_columns: list[str], max_depth: int, class_weight: Optional[str], seed: int) -> dict:
    X = df[feature_columns]
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed, stratify=y)
    clf = make_classifier(max_depth=max_depth, class_weight=class_weight, seed=seed)
    clf.fit(X_train, y_train)
    y_train_pred = clf.predict(X_train)
    y_pred = clf.predict(X_test)

    cm = confusion_matrix(y_test, y_pred, labels=PHASE_ORDER)
    results = X_test.copy()
    results["true"] = y_test.values
    results["pred"] = y_pred
    for col in DIAGNOSTIC_COLUMNS:
        if col in df.columns and col not in results.columns:
            results[col] = df.loc[X_test.index, col].values
    errors = results[results["true"] != results["pred"]].copy()
    categorized_errors = categorize_errors(errors)

    importances = pd.DataFrame({
        "feature_set": feature_set_name,
        "feature": feature_columns,
        "importance": clf.feature_importances_,
        "seed": seed,
        "max_depth": max_depth,
        "class_weight": str(class_weight),
    }).sort_values("importance", ascending=False)

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
        "total_errors": int((y_pred != y_test).sum()),
        "importances": importances,
        "categorized_errors": categorized_errors,
        "confusion_matrix": cm,
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
                    raw_rows.append({
                        "feature_set": feature_set_name,
                        "seed": seed,
                        "max_depth": depth,
                        "class_weight": str(class_weight),
                        "train_accuracy": result["train_accuracy"],
                        "test_accuracy": result["test_accuracy"],
                        "accuracy_gap": result["train_accuracy"] - result["test_accuracy"],
                        "balanced_accuracy": result["balanced_accuracy"],
                        "macro_f1": result["macro_f1"],
                        "weighted_f1": result["weighted_f1"],
                        "actual_depth": result["actual_depth"],
                        "number_of_leaves": result["number_of_leaves"],
                        "total_errors": result["total_errors"],
                        "top_feature": result["importances"].iloc[0]["feature"],
                    })
                    importances.append(result["importances"])

                    if not result["categorized_errors"].empty:
                        counts = result["categorized_errors"]["primary_error_category"].value_counts().reset_index()
                        counts.columns = ["primary_error_category", "count"]
                        for _, row in counts.iterrows():
                            error_category_rows.append({
                                "feature_set": feature_set_name,
                                "seed": seed,
                                "max_depth": depth,
                                "class_weight": str(class_weight),
                                "primary_error_category": row["primary_error_category"],
                                "count": int(row["count"]),
                            })

                    print(
                        f"seed={seed} | depth={depth} | class_weight={class_weight} | "
                        f"acc={result['test_accuracy']:.4f} | bal_acc={result['balanced_accuracy']:.4f} | "
                        f"errors={result['total_errors']} | leaves={result['number_of_leaves']}"
                    )

    return pd.DataFrame(raw_rows), pd.concat(importances, ignore_index=True), pd.DataFrame(error_category_rows)


def summarize_results(raw_results: pd.DataFrame) -> pd.DataFrame:
    return raw_results.groupby(["feature_set", "max_depth", "class_weight"]).agg(
        test_accuracy_mean=("test_accuracy", "mean"),
        test_accuracy_std=("test_accuracy", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        balanced_accuracy_std=("balanced_accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        total_errors_mean=("total_errors", "mean"),
        total_errors_std=("total_errors", "std"),
        number_of_leaves_mean=("number_of_leaves", "mean"),
        number_of_leaves_std=("number_of_leaves", "std"),
    ).reset_index()


def create_main_table(summary: pd.DataFrame, main_depth: int, main_class_weight: Optional[str]) -> pd.DataFrame:
    main_class_weight_str = normalize_class_weight_value(main_class_weight)
    main = summary[
        (summary["max_depth"] == main_depth)
        & (summary["class_weight"].apply(normalize_class_weight_value) == main_class_weight_str)
    ].copy()
    order = {"raw_coordinates": 1, "geometric_features": 2, "full_kqk_strategic_features": 3}
    main["feature_set_order"] = main["feature_set"].map(order)
    main = main.sort_values("feature_set_order").drop(columns=["feature_set_order"])
    if main.empty:
        return main
    raw_row = main[main["feature_set"] == "raw_coordinates"]
    geo_row = main[main["feature_set"] == "geometric_features"]
    if not raw_row.empty:
        raw_errors = float(raw_row.iloc[0]["total_errors_mean"])
        raw_bal = float(raw_row.iloc[0]["balanced_accuracy_mean"])
        raw_macro_f1 = float(raw_row.iloc[0]["macro_f1_mean"])

        main["error_reduction_vs_raw"] = raw_errors - main["total_errors_mean"]
        main["balanced_accuracy_gain_vs_raw"] = main["balanced_accuracy_mean"] - raw_bal
        main["macro_f1_gain_vs_raw"] = main["macro_f1_mean"] - raw_macro_f1

    if not geo_row.empty:
        geo_errors = float(geo_row.iloc[0]["total_errors_mean"])
        geo_bal = float(geo_row.iloc[0]["balanced_accuracy_mean"])
        geo_macro_f1 = float(geo_row.iloc[0]["macro_f1_mean"])

        main["error_reduction_vs_geometric"] = geo_errors - main["total_errors_mean"]
        main["balanced_accuracy_gain_vs_geometric"] = main["balanced_accuracy_mean"] - geo_bal
        main["macro_f1_gain_vs_geometric"] = main["macro_f1_mean"] - geo_macro_f1
    return main


def summarize_importances(importances: pd.DataFrame) -> pd.DataFrame:
    return importances.groupby(["feature_set", "feature"]).agg(
        importance_mean=("importance", "mean"),
        importance_std=("importance", "std"),
        importance_min=("importance", "min"),
        importance_max=("importance", "max"),
    ).reset_index().sort_values(["feature_set", "importance_mean"], ascending=[True, False])


def summarize_error_categories(error_categories: pd.DataFrame) -> pd.DataFrame:
    if error_categories.empty:
        return pd.DataFrame()
    return error_categories.groupby(["feature_set", "max_depth", "class_weight", "primary_error_category"]).agg(
        count_mean=("count", "mean"),
        count_std=("count", "std"),
        count_min=("count", "min"),
        count_max=("count", "max"),
    ).reset_index().sort_values(["feature_set", "max_depth", "class_weight", "count_mean"], ascending=[True, True, True, False])


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print_section("KQK FEATURE SET COMPARISON")
    print(f"Dataset: {DATASET_CSV}")
    print(f"Output folder: {COMPARISON_DIR}")

    print_section("RESOLVING MAIN COMPARISON CONFIG")
    main_depth, main_class_weight, source_description = load_main_config_from_dectree()
    main_class_weight_str = normalize_class_weight_value(main_class_weight)
    print(f"\nMain comparison configuration source:\n  {source_description}")

    df = read_csv_auto(DATASET_CSV)
    validate_feature_sets(df)

    print(f"\nRows: {len(df):,}")
    print(f"Target column: {TARGET_COLUMN}")
    print("\nTarget distribution:")
    print(df[TARGET_COLUMN].value_counts())

    raw_results, raw_importances, raw_error_categories = run_comparison(df)
    stability_summary = summarize_results(raw_results)
    main_comparison = create_main_table(stability_summary, main_depth, main_class_weight)
    mean_importances = summarize_importances(raw_importances)
    error_category_summary = summarize_error_categories(raw_error_categories)

    print_section(f"MAIN COMPARISON @ depth={main_depth}, class_weight={main_class_weight_str}")
    if main_comparison.empty:
        print(f"\n[WARNING] No stability rows for depth={main_depth}, class_weight={main_class_weight_str}.")
    else:
        print(main_comparison.to_string(index=False))

    config_used_df = pd.DataFrame([{
        "main_depth": main_depth,
        "main_class_weight": str(main_class_weight),
        "source": source_description,
    }])

    main_table_filename = f"{OUTPUT_PREFIX}_main_depth{main_depth}_{main_class_weight_str.lower()}_comparison.csv"

    save_csv(raw_results, COMPARISON_DIR / f"{OUTPUT_PREFIX}_raw_results.csv")
    save_csv(stability_summary, COMPARISON_DIR / f"{OUTPUT_PREFIX}_stability_summary.csv")
    save_csv(main_comparison, COMPARISON_DIR / main_table_filename)
    save_csv(config_used_df, COMPARISON_DIR / f"{OUTPUT_PREFIX}_main_config_used.csv")
    save_csv(raw_importances, COMPARISON_DIR / f"{OUTPUT_PREFIX}_raw_importances.csv")
    save_csv(mean_importances, COMPARISON_DIR / f"{OUTPUT_PREFIX}_mean_importances.csv")
    save_csv(raw_error_categories, COMPARISON_DIR / f"{OUTPUT_PREFIX}_error_categories_raw.csv")
    save_csv(error_category_summary, COMPARISON_DIR / f"{OUTPUT_PREFIX}_error_category_summary.csv")

    print_section("FILES SAVED")
    for path in sorted(COMPARISON_DIR.glob(f"{OUTPUT_PREFIX}_*")):
        print(path)


if __name__ == "__main__":
    main()
