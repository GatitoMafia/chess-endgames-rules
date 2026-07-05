from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

KPK_OUTPUT_ROOT = Path(r"C:\Users\Κωνσταντίνος\Desktop\KPK Dataset")

MODEL_DIR = KPK_OUTPUT_ROOT / "02_model_wdl"
CONSOLIDATION_DIR = KPK_OUTPUT_ROOT / "05_rule_consolidation"
CONSOLIDATION_DIR.mkdir(parents=True, exist_ok=True)

RULES_CSV = MODEL_DIR / "kpk_wdl_decision_tree_all_leaf_rules.csv"

OUTPUT_PREFIX = "kpk_wdl_rule_consolidation"

CSV_SEPARATOR = ";"
CSV_ENCODING = "utf-8-sig"
DECIMAL_PLACES = 4
FLOAT_FORMAT = f"%.{DECIMAL_PLACES}f"

pd.set_option("display.float_format", lambda value: f"{value:.{DECIMAL_PLACES}f}")


# ============================================================
# KPK FEATURE ORDER / BINNING
# ============================================================

RESULT_ORDER = ["Draw", "Win"]

FEATURE_ORDER = [
    "side_to_move",
    "steps_to_promotion",
    "is_rook_pawn",
    "black_king_ahead_of_pawn",
    "white_king_pawn_distance",
    "black_king_pawn_distance",
    "pawn_distance_diff",
    "kings_distance",
    "black_king_inside_square_of_pawn",
    "black_king_can_block_promotion",
    "white_wins_key_square_race",
    "kings_have_direct_opposition",
    "kings_have_distant_opposition",
    "white_king_strongly_supports_pawn",
]

BINARY_FEATURE_LABELS = {
    "side_to_move": {0: "BLACK_TO_MOVE", 1: "WHITE_TO_MOVE"},
    "is_rook_pawn": {0: "NO", 1: "YES"},
    "black_king_ahead_of_pawn": {0: "NO", 1: "YES"},
    "black_king_inside_square_of_pawn": {0: "NO", 1: "YES"},
    "black_king_can_block_promotion": {0: "NO", 1: "YES"},
    "white_wins_key_square_race": {0: "NO", 1: "YES"},
    "kings_have_direct_opposition": {0: "NO", 1: "YES"},
    "kings_have_distant_opposition": {0: "NO", 1: "YES"},
    "white_king_strongly_supports_pawn": {0: "NO", 1: "YES"},
}

NUMERIC_BINS = {
    "steps_to_promotion": [
        (0, 2, "VERY_ADVANCED"),
        (3, 4, "ADVANCED"),
        (5, 6, "FAR"),
    ],
    "white_king_pawn_distance": [
        (0, 1, "VERY_CLOSE"),
        (2, 2, "CLOSE"),
        (3, 4, "MEDIUM"),
        (5, 7, "FAR"),
    ],
    "black_king_pawn_distance": [
        (0, 1, "VERY_CLOSE"),
        (2, 2, "CLOSE"),
        (3, 4, "MEDIUM"),
        (5, 7, "FAR"),
    ],
    "pawn_distance_diff": [
        (-7, -2, "BLACK_CLOSER"),
        (-1, 1, "SIMILAR"),
        (2, 7, "WHITE_CLOSER"),
    ],
    "kings_distance": [
        (0, 2, "CLOSE"),
        (3, 4, "MEDIUM"),
        (5, 7, "FAR"),
    ],
}


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


def normalize_result(value: Any) -> str:
    value_str = str(value).strip()

    if value_str in {"0", "0.0"}:
        return "Draw"

    if value_str in {"1", "1.0"}:
        return "Win"

    if value_str.lower() == "draw":
        return "Draw"

    if value_str.lower() == "win":
        return "Win"

    return value_str


def result_rank(result: str) -> int:
    try:
        return RESULT_ORDER.index(result)
    except ValueError:
        return len(RESULT_ORDER)


def safe_numeric(value: Any, default: float = 0.0) -> float:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return default
    return float(numeric)


# ============================================================
# RULE PARSING / BINNING
# ============================================================

def parse_condition(condition: str):
    match = re.match(
        r"^\s*([A-Za-z0-9_]+)\s*(<=|>)\s*(-?\d+(?:\.\d+)?)\s*$",
        condition,
    )

    if match is None:
        return None

    feature = match.group(1)
    operator = match.group(2)
    threshold = float(match.group(3))

    return feature, operator, threshold


def update_range_from_condition(
    current_range: tuple[float, float],
    operator: str,
    threshold: float,
) -> tuple[float, float]:
    low, high = current_range

    if operator == "<=":
        high = min(high, threshold)
    else:
        low = max(low, threshold)

    return low, high


def extract_feature_ranges(technical_rule: str) -> dict[str, tuple[float, float]]:
    feature_ranges: dict[str, tuple[float, float]] = {
        feature: (-float("inf"), float("inf"))
        for feature in FEATURE_ORDER
    }

    if not isinstance(technical_rule, str) or technical_rule.strip() in {"", "ROOT"}:
        return feature_ranges

    for condition in technical_rule.split(" AND "):
        parsed = parse_condition(condition)

        if parsed is None:
            continue

        feature, operator, threshold = parsed

        if feature not in feature_ranges:
            continue

        feature_ranges[feature] = update_range_from_condition(
            current_range=feature_ranges[feature],
            operator=operator,
            threshold=threshold,
        )

    return feature_ranges


def infer_integer_value_from_range(value_range: tuple[float, float]) -> int | None:
    low, high = value_range

    possible_values = []

    for value in range(-10, 11):
        if value > low and value <= high:
            possible_values.append(value)

    if len(possible_values) == 1:
        return possible_values[0]

    return None


def infer_binary_value(value_range: tuple[float, float]) -> int | None:
    low, high = value_range

    possible_values = []

    for value in (0, 1):
        if value > low and value <= high:
            possible_values.append(value)

    if len(possible_values) == 1:
        return possible_values[0]

    return None


def describe_numeric_range(
    feature: str,
    value_range: tuple[float, float],
) -> str:
    integer_value = infer_integer_value_from_range(value_range)

    if integer_value is not None:
        for low, high, label in NUMERIC_BINS.get(feature, []):
            if low <= integer_value <= high:
                return label

        return str(integer_value)

    low, high = value_range

    if feature == "steps_to_promotion":
        if high <= 2.5:
            return "VERY_ADVANCED"
        if high <= 4.5:
            return "ADVANCED"
        if low > 4.5:
            return "FAR"
        return "MIXED_DISTANCE"

    if feature in {"white_king_pawn_distance", "black_king_pawn_distance", "kings_distance"}:
        if high <= 1.5:
            return "VERY_CLOSE"
        if high <= 2.5:
            return "CLOSE"
        if high <= 4.5:
            return "MEDIUM"
        if low > 4.5:
            return "FAR"
        return "MIXED_DISTANCE"

    if feature == "pawn_distance_diff":
        if high <= -1.5:
            return "BLACK_CLOSER"
        if low > 1.5:
            return "WHITE_CLOSER"
        if low > -1.5 and high <= 1.5:
            return "SIMILAR"
        return "MIXED_DIFF"

    return "ANY"


def describe_feature_bin(
    feature: str,
    value_range: tuple[float, float],
) -> str:
    if feature in BINARY_FEATURE_LABELS:
        value = infer_binary_value(value_range)

        if value is None:
            return "ANY"

        return BINARY_FEATURE_LABELS[feature][value]

    return describe_numeric_range(feature, value_range)


def build_binned_signature(technical_rule: str) -> tuple[str, dict[str, str]]:
    feature_ranges = extract_feature_ranges(technical_rule)

    feature_bins = {
        feature: describe_feature_bin(feature, feature_ranges[feature])
        for feature in FEATURE_ORDER
    }

    signature_parts = [
        f"{feature}={feature_bins[feature]}"
        for feature in FEATURE_ORDER
        if feature_bins[feature] != "ANY"
    ]

    if not signature_parts:
        return "UNSPECIFIED", feature_bins

    return " | ".join(signature_parts), feature_bins


# ============================================================
# INTERPRETIVE MOTIFS
# ============================================================

def identify_interpretive_motif(
    predicted_result: str,
    feature_bins: dict[str, str],
) -> tuple[str, str]:
    side_to_move = feature_bins.get("side_to_move", "ANY")
    steps = feature_bins.get("steps_to_promotion", "ANY")
    rook_pawn = feature_bins.get("is_rook_pawn", "ANY")
    black_ahead = feature_bins.get("black_king_ahead_of_pawn", "ANY")
    black_inside_square = feature_bins.get("black_king_inside_square_of_pawn", "ANY")
    black_can_block = feature_bins.get("black_king_can_block_promotion", "ANY")
    white_wins_key_race = feature_bins.get("white_wins_key_square_race", "ANY")
    direct_opposition = feature_bins.get("kings_have_direct_opposition", "ANY")
    distant_opposition = feature_bins.get("kings_have_distant_opposition", "ANY")
    strong_support = feature_bins.get("white_king_strongly_supports_pawn", "ANY")
    pawn_distance_diff = feature_bins.get("pawn_distance_diff", "ANY")

    has_opposition = direct_opposition == "YES" or distant_opposition == "YES"

    if predicted_result == "Win":
        if black_inside_square == "NO" and black_can_block == "NO":
            return (
                "win_promotion_race_signature",
                "Νίκη: ο μαύρος βασιλιάς δεν βρίσκεται μέσα στο τετράγωνο του πιονιού και δεν προλαβαίνει να δημιουργήσει blockade.",
            )

        if white_wins_key_race == "YES" and strong_support == "YES":
            return (
                "win_key_square_and_support_signature",
                "Νίκη: ο λευκός κερδίζει τα κρίσιμα τετράγωνα και ο βασιλιάς του στηρίζει ενεργά το πιόνι.",
            )

        if black_inside_square == "NO":
            return (
                "win_black_outside_square_signature",
                "Νίκη: ο μαύρος βασιλιάς βρίσκεται εκτός του τετραγώνου του πιονιού, άρα δεν προλαβαίνει εύκολα την προαγωγή.",
            )

        if black_can_block == "NO":
            return (
                "win_no_blockade_signature",
                "Νίκη: ο μαύρος βασιλιάς δεν μπορεί να οργανώσει έγκαιρο blockade απέναντι στην προαγωγή.",
            )

        if white_wins_key_race == "YES":
            return (
                "win_key_square_race_only_signature",
                "Νίκη: ο λευκός βασιλιάς κερδίζει τον αγώνα προς τα κρίσιμα τετράγωνα.",
            )

        if steps == "VERY_ADVANCED" and rook_pawn == "NO":
            return (
                "win_advanced_non_rook_pawn_signature",
                "Νίκη: το πιόνι είναι πολύ προχωρημένο και δεν είναι rook pawn, άρα οι πιθανότητες μετατροπής είναι αυξημένες.",
            )

        if side_to_move == "WHITE_TO_MOVE" and black_can_block != "YES":
            return (
                "win_white_to_move_tempo_signature",
                "Νίκη: ο λευκός έχει τη σειρά και δεν υπάρχει σαφές άμεσο blockade από τον μαύρο.",
            )

        return (
            "win_other_data_signature",
            "Νίκη: το leaf αντιστοιχεί σε κερδισμένες θέσεις χωρίς να κυριαρχεί ένα από τα κύρια προκαθορισμένα KPK motifs.",
        )

    if predicted_result == "Draw":
        if rook_pawn == "YES" and strong_support != "YES":
            return (
                "draw_rook_pawn_exception_signature",
                "Ισοπαλία: το rook pawn σε συνδυασμό με ανεπαρκή στήριξη του λευκού βασιλιά ευνοεί την άμυνα.",
            )

        if black_inside_square == "YES" and black_can_block == "YES":
            return (
                "draw_square_rule_and_blockade_signature",
                "Ισοπαλία: ο μαύρος βρίσκεται μέσα στο τετράγωνο του πιονιού και μπορεί να οργανώσει blockade.",
            )

        if black_can_block == "YES":
            return (
                "draw_promotion_blockade_signature",
                "Ισοπαλία: ο μαύρος βασιλιάς προλαβαίνει να μπλοκάρει την προαγωγή.",
            )

        if black_inside_square == "YES":
            return (
                "draw_square_rule_signature",
                "Ισοπαλία: ο μαύρος βασιλιάς βρίσκεται μέσα στο τετράγωνο του πιονιού.",
            )

        if black_ahead == "YES":
            return (
                "draw_black_king_ahead_signature",
                "Ισοπαλία: ο μαύρος βασιλιάς βρίσκεται μπροστά από το πιόνι και μπορεί να καθυστερήσει ή να μπλοκάρει την πρόοδο.",
            )

        if has_opposition:
            return (
                "draw_opposition_tempo_signature",
                "Ισοπαλία: το opposition/tempo motif λειτουργεί αμυντικά και δεν επιτρέπει καθαρή μετατροπή.",
            )

        return (
            "draw_other_data_signature",
            "Ισοπαλία: το leaf αντιστοιχεί σε ισόπαλες θέσεις χωρίς να κυριαρχεί ένα από τα κύρια προκαθορισμένα KPK motifs.",
        )

    return (
        "unknown_result_signature",
        "Δεν αναγνωρίστηκε η προβλεπόμενη κλάση του leaf.",
    )


# ============================================================
# CONSOLIDATION TABLES
# ============================================================

def enrich_leaves_with_signatures(rules_df: pd.DataFrame) -> pd.DataFrame:
    leaves = rules_df.copy()

    if "predicted_result" not in leaves.columns:
        if "predicted_label" not in leaves.columns:
            raise ValueError("Rules file must contain either predicted_result or predicted_label.")

        leaves["predicted_result"] = leaves["predicted_label"].apply(normalize_result)

    leaves["predicted_result"] = leaves["predicted_result"].apply(normalize_result)

    signatures = []
    motif_names = []
    motif_descriptions = []
    feature_bin_rows = []

    for _, row in leaves.iterrows():
        technical_rule = str(row.get("technical_rule", "ROOT"))
        signature, feature_bins = build_binned_signature(technical_rule)

        motif_name, motif_description = identify_interpretive_motif(
            predicted_result=str(row["predicted_result"]),
            feature_bins=feature_bins,
        )

        signatures.append(signature)
        motif_names.append(motif_name)
        motif_descriptions.append(motif_description)
        feature_bin_rows.append(feature_bins)

    leaves["binned_signature"] = signatures
    leaves["signature_key"] = (
        leaves["predicted_result"].astype(str)
        + " | "
        + leaves["binned_signature"].astype(str)
    )
    leaves["interpretive_motif"] = motif_names
    leaves["interpretive_description"] = motif_descriptions

    for feature in FEATURE_ORDER:
        leaves[f"{feature}_bin"] = [
            feature_bins.get(feature, "ANY")
            for feature_bins in feature_bin_rows
        ]

    if "samples" in leaves.columns:
        leaves["samples"] = pd.to_numeric(leaves["samples"], errors="coerce").fillna(0).astype(int)
    else:
        leaves["samples"] = 0

    if "purity" in leaves.columns:
        leaves["purity"] = pd.to_numeric(leaves["purity"], errors="coerce")
    else:
        leaves["purity"] = 0.0

    return leaves


def build_distinct_signatures(leaves: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        leaves
        .groupby(["predicted_result", "binned_signature", "signature_key", "interpretive_motif", "interpretive_description"])
        .agg(
            n_leaves=("signature_key", "size"),
            total_samples=("samples", "sum"),
            mean_samples=("samples", "mean"),
            mean_purity=("purity", "mean"),
            min_purity=("purity", "min"),
            max_purity=("purity", "max"),
        )
        .reset_index()
    )

    result_totals = (
        grouped
        .groupby("predicted_result")["total_samples"]
        .sum()
        .reset_index(name="total_samples_by_result")
    )

    grouped = grouped.merge(result_totals, on="predicted_result", how="left")

    grouped["coverage_pct"] = (
        100 * grouped["total_samples"] / grouped["total_samples_by_result"]
    )

    grouped["result_rank"] = grouped["predicted_result"].apply(result_rank)

    grouped = grouped.sort_values(
        ["result_rank", "total_samples", "n_leaves", "mean_purity"],
        ascending=[True, False, False, False],
    ).drop(columns=["result_rank"])

    return grouped


def build_top_signatures_by_result(
    distinct_signatures: pd.DataFrame,
    top_n: int = 15,
) -> pd.DataFrame:
    if distinct_signatures.empty:
        return distinct_signatures

    top = (
        distinct_signatures
        .sort_values(["predicted_result", "total_samples"], ascending=[True, False])
        .groupby("predicted_result", group_keys=False)
        .head(top_n)
        .copy()
    )

    top["result_rank"] = top["predicted_result"].apply(result_rank)

    return top.sort_values(
        ["result_rank", "total_samples"],
        ascending=[True, False],
    ).drop(columns=["result_rank"])


def build_interpretive_motifs(leaves: pd.DataFrame) -> pd.DataFrame:
    motifs = (
        leaves
        .groupby(["predicted_result", "interpretive_motif", "interpretive_description"])
        .agg(
            n_signatures=("signature_key", "nunique"),
            n_leaves=("interpretive_motif", "size"),
            total_samples=("samples", "sum"),
            mean_samples=("samples", "mean"),
            mean_purity=("purity", "mean"),
            min_purity=("purity", "min"),
            max_purity=("purity", "max"),
        )
        .reset_index()
    )

    result_totals = (
        motifs
        .groupby("predicted_result")["total_samples"]
        .sum()
        .reset_index(name="total_samples_by_result")
    )

    motifs = motifs.merge(result_totals, on="predicted_result", how="left")

    motifs["coverage_pct"] = (
        100 * motifs["total_samples"] / motifs["total_samples_by_result"]
    )

    motifs["result_rank"] = motifs["predicted_result"].apply(result_rank)

    motifs = motifs.sort_values(
        ["result_rank", "total_samples", "n_leaves"],
        ascending=[True, False, False],
    ).drop(columns=["result_rank"])

    return motifs


def build_result_coverage(leaves: pd.DataFrame) -> pd.DataFrame:
    coverage = (
        leaves
        .groupby("predicted_result")
        .agg(
            n_signatures=("signature_key", "nunique"),
            n_leaves=("predicted_result", "size"),
            total_samples=("samples", "sum"),
            mean_purity=("purity", "mean"),
            min_purity=("purity", "min"),
            max_purity=("purity", "max"),
            n_motifs=("interpretive_motif", "nunique"),
        )
        .reset_index()
    )

    total_samples_all = coverage["total_samples"].sum()
    total_leaves_all = coverage["n_leaves"].sum()

    if total_samples_all > 0:
        coverage["coverage_pct"] = 100 * coverage["total_samples"] / total_samples_all
    else:
        coverage["coverage_pct"] = 0.0

    if total_leaves_all > 0:
        coverage["leaf_coverage_pct"] = 100 * coverage["n_leaves"] / total_leaves_all
    else:
        coverage["leaf_coverage_pct"] = 0.0

    coverage["result_rank"] = coverage["predicted_result"].apply(result_rank)

    return coverage.sort_values("result_rank").drop(columns=["result_rank"])


# ============================================================
# SUMMARY MARKDOWN
# ============================================================

def write_summary_markdown(
    leaves: pd.DataFrame,
    distinct_signatures: pd.DataFrame,
    top_signatures: pd.DataFrame,
    motifs: pd.DataFrame,
    result_coverage: pd.DataFrame,
) -> None:
    summary_path = CONSOLIDATION_DIR / f"{OUTPUT_PREFIX}_summary.md"

    total_leaves = int(len(leaves))
    total_samples = int(leaves["samples"].sum()) if "samples" in leaves.columns else 0
    total_signatures = int(len(distinct_signatures))
    total_motifs = int(len(motifs))

    coverage_text = result_coverage.to_string(index=False) if not result_coverage.empty else "No result coverage available."

    top_signature_cols = [
        "predicted_result",
        "interpretive_motif",
        "total_samples",
        "n_leaves",
        "mean_purity",
        "coverage_pct",
        "binned_signature",
    ]

    top_existing_cols = [col for col in top_signature_cols if col in top_signatures.columns]

    if top_signatures.empty:
        top_text = "No top signatures available."
    else:
        top_text = top_signatures[top_existing_cols].head(20).to_string(index=False)

    motif_cols = [
        "predicted_result",
        "interpretive_motif",
        "total_samples",
        "n_leaves",
        "n_signatures",
        "mean_purity",
        "coverage_pct",
    ]

    motif_existing_cols = [col for col in motif_cols if col in motifs.columns]

    if motifs.empty:
        motif_text = "No motifs available."
    else:
        motif_text = motifs[motif_existing_cols].to_string(index=False)

    summary = f"""# KPK WDL Rule Consolidation Summary

## 1. Purpose

This file summarizes the consolidation of KPK WDL decision-tree leaf rules into broader data-driven signatures and chess-oriented interpretive motifs.

Unlike KRK and KQK, the KPK task is not DTZ-phase classification. It is a WDL binary classification task:

- `Draw`
- `Win`

Therefore, consolidation is performed by grouping leaves according to:

predicted_result + binned_signature

## 2. Overall Counts

| Quantity | Value |
|---|---:|
| Total leaves | {total_leaves} |
| Total leaf samples | {total_samples} |
| Distinct signatures | {total_signatures} |
| Interpretive motifs | {total_motifs} |

## 3. Result Coverage

{coverage_text}

## 4. Main KPK Motifs

The interpretive motifs are based on KPK-specific concepts:

- square of the pawn,
- promotion blockade,
- key-square race,
- rook-pawn exception,
- opposition / tempo,
- strong king support.

These motifs are diagnostic summaries of learned decision-tree leaves. They are not formal chess theorems.

## 5. Top Signatures

{top_text}

## 6. Interpretive Motif Coverage

{motif_text}

## 7. Output Files

The rule consolidation module writes:

- `{OUTPUT_PREFIX}_leaves_with_signature.csv`
- `{OUTPUT_PREFIX}_distinct_signatures.csv`
- `{OUTPUT_PREFIX}_top_signatures_by_result.csv`
- `{OUTPUT_PREFIX}_interpretive_motifs.csv`
- `{OUTPUT_PREFIX}_result_coverage.csv`
- `{OUTPUT_PREFIX}_summary.md`
"""

    summary_path.write_text(summary, encoding="utf-8")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print_section("KPK WDL RULE CONSOLIDATION")

    print(f"Input rules CSV: {RULES_CSV}")
    print(f"Output folder:   {CONSOLIDATION_DIR}")

    rules_df = read_csv_auto(RULES_CSV)

    if rules_df.empty:
        raise ValueError(f"Rules file is empty: {RULES_CSV}")

    print(f"\nLoaded leaf rules: {len(rules_df):,}")
    print("Columns:")
    print(rules_df.columns.tolist())

    leaves = enrich_leaves_with_signatures(rules_df)
    distinct_signatures = build_distinct_signatures(leaves)
    top_signatures = build_top_signatures_by_result(distinct_signatures)
    motifs = build_interpretive_motifs(leaves)
    result_coverage = build_result_coverage(leaves)

    save_csv(
        leaves,
        CONSOLIDATION_DIR / f"{OUTPUT_PREFIX}_leaves_with_signature.csv",
    )

    save_csv(
        distinct_signatures,
        CONSOLIDATION_DIR / f"{OUTPUT_PREFIX}_distinct_signatures.csv",
    )

    save_csv(
        top_signatures,
        CONSOLIDATION_DIR / f"{OUTPUT_PREFIX}_top_signatures_by_result.csv",
    )

    save_csv(
        motifs,
        CONSOLIDATION_DIR / f"{OUTPUT_PREFIX}_interpretive_motifs.csv",
    )

    save_csv(
        result_coverage,
        CONSOLIDATION_DIR / f"{OUTPUT_PREFIX}_result_coverage.csv",
    )

    write_summary_markdown(
        leaves=leaves,
        distinct_signatures=distinct_signatures,
        top_signatures=top_signatures,
        motifs=motifs,
        result_coverage=result_coverage,
    )

    print_section("CONSOLIDATION SUMMARY")

    print("\nResult coverage:")
    print(result_coverage.to_string(index=False))

    print("\nTop interpretive motifs:")
    if motifs.empty:
        print("No motifs available.")
    else:
        print(
            motifs[
                [
                    "predicted_result",
                    "interpretive_motif",
                    "total_samples",
                    "n_leaves",
                    "n_signatures",
                    "mean_purity",
                    "coverage_pct",
                ]
            ].to_string(index=False)
        )

    print_section("FILES SAVED")

    for path in sorted(CONSOLIDATION_DIR.glob(f"{OUTPUT_PREFIX}_*")):
        print(path)


if __name__ == "__main__":
    main()
