from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Optional

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

KQK_OUTPUT_ROOT = Path(r"C:\Users\Κωνσταντίνος\Desktop\KQK Dataset")

MODEL_DIR = KQK_OUTPUT_ROOT / "02_model_dtz"
CONSOLIDATION_DIR = KQK_OUTPUT_ROOT / "05_rule_consolidation"

ALL_LEAF_RULES_CSV = MODEL_DIR / "kqk_dtz_decision_tree_all_leaf_rules.csv"

OUTPUT_PREFIX = "kqk_rule_consolidation"

CSV_SEPARATOR = ";"
DECIMAL_PLACES = 4
FLOAT_FORMAT = f"%.{DECIMAL_PLACES}f"

TOP_N_SIGNATURES_PER_PHASE = 15

PHASE_ORDER = [
    "long_conversion",
    "medium_conversion",
    "short_conversion",
    "immediate_mate",
]

PHASE_DISPLAY_NAMES = {
    "long_conversion": "Long conversion",
    "medium_conversion": "Medium conversion",
    "short_conversion": "Short conversion",
    "immediate_mate": "Immediate mate",
}

PHASE_SORT_ORDER = {phase: idx for idx, phase in enumerate(PHASE_ORDER)}


# ============================================================
# BINNING DEFINITIONS
# ============================================================

NUMERIC_BINS = {
    "bk_legal_moves_if_black_to_move": [
        ("VERY_LOW", 1),
        ("LOW", 2),
        ("MEDIUM", 4),
        ("HIGH", 99),
    ],
    "bk_distance_to_corner": [
        ("AT_CORNER", 0),
        ("NEAR_CORNER", 2),
        ("FAR_FROM_CORNER", 99),
    ],
    "bk_distance_to_edge": [
        ("ON_EDGE", 0),
        ("NEAR_EDGE", 1),
        ("FAR_FROM_EDGE", 99),
    ],
    "wk_distance_to_edge": [
        ("ON_EDGE", 0),
        ("NEAR_EDGE", 2),
        ("FAR_FROM_EDGE", 99),
    ],
    "wk_wq_distance": [
        ("CLOSE", 2),
        ("MEDIUM", 4),
        ("FAR", 99),
    ],
    "wk_bk_chebyshev_distance": [
        ("CLOSE", 2),
        ("MEDIUM", 3),
        ("FAR", 99),
    ],
    "wk_bk_manhattan_distance": [
        ("CLOSE", 3),
        ("MEDIUM", 5),
        ("FAR", 99),
    ],
    "wq_bk_chebyshev_distance": [
        ("CLOSE", 2),
        ("MEDIUM", 4),
        ("FAR", 99),
    ],
    "queen_box_area": [
        ("TINY", 8),
        ("SMALL", 16),
        ("MEDIUM", 32),
        ("LARGE", 99),
    ],
    "queen_controls_bk_zone_count": [
        ("LOW", 2),
        ("MEDIUM", 4),
        ("HIGH", 8),
    ],
    "stalemate_risk_level": [
        ("NONE", 0),
        ("MODERATE", 1),
        ("CRITICAL", 2),
    ],
}

FEATURE_ORDER = [
    "bk_legal_moves_if_black_to_move",
    "bk_distance_to_edge",
    "bk_distance_to_corner",
    "wk_bk_chebyshev_distance",
    "wk_bk_manhattan_distance",
    "wk_distance_to_edge",
    "wk_wq_distance",
    "wq_bk_chebyshev_distance",
    "queen_box_area",
    "queen_controls_bk_zone_count",
    "stalemate_risk_level",
]


# ============================================================
# BASIC UTILITIES
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


def save_csv(df: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    output = df.copy()

    if "predicted_phase_display" in output.columns:
        output = output.drop(columns=["predicted_phase_display"])

    path.parent.mkdir(parents=True, exist_ok=True)

    output.to_csv(
        path,
        index=index,
        encoding="utf-8-sig",
        sep=CSV_SEPARATOR,
        float_format=FLOAT_FORMAT,
        decimal=",",
    )


def phase_sort_key(phase: str) -> int:
    return PHASE_SORT_ORDER.get(str(phase), 999)


def safe_to_numeric(value, default: float = 0.0) -> float:
    result = pd.to_numeric(value, errors="coerce")

    if pd.isna(result):
        return default

    return float(result)


def parse_class_distribution(value) -> dict:
    if isinstance(value, dict):
        return value

    if not isinstance(value, str) or not value.strip():
        return {}

    try:
        parsed = ast.literal_eval(value)
    except Exception:
        return {}

    if isinstance(parsed, dict):
        return parsed

    return {}


# ============================================================
# CONDITION PARSING
# ============================================================

CONDITION_PATTERN = re.compile(
    r"^\s*([A-Za-z0-9_]+)\s*(<=|>)\s*(-?\d+(?:\.\d+)?)\s*$"
)


def parse_condition(condition: str) -> Optional[tuple[str, str, float]]:
    match = CONDITION_PATTERN.match(str(condition).strip())

    if match is None:
        return None

    feature = match.group(1)
    operator = match.group(2)
    threshold = float(match.group(3))

    return feature, operator, threshold


def split_technical_rule(technical_rule: str) -> list[str]:
    if not isinstance(technical_rule, str):
        return []

    technical_rule = technical_rule.strip()

    if not technical_rule or technical_rule == "ROOT":
        return []

    return [part.strip() for part in technical_rule.split(" AND ") if part.strip()]


# ============================================================
# SIGNATURE BINNING
# ============================================================

def build_numeric_intervals(technical_rule: str) -> dict[str, dict[str, Optional[int]]]:
    """
    Convert a decision-tree path into integer intervals per feature.

    Example:
    queen_box_area > 16.50 AND queen_box_area <= 32.50

    becomes:
    queen_box_area: lower=17, upper=32
    """

    intervals: dict[str, dict[str, Optional[int]]] = {}

    for condition in split_technical_rule(technical_rule):
        parsed = parse_condition(condition)

        if parsed is None:
            continue

        feature, operator, threshold = parsed

        if feature not in NUMERIC_BINS:
            continue

        intervals.setdefault(feature, {"lower": None, "upper": None})

        if operator == "<=":
            upper_value = int(threshold)
            current_upper = intervals[feature]["upper"]

            if current_upper is None:
                intervals[feature]["upper"] = upper_value
            else:
                intervals[feature]["upper"] = min(current_upper, upper_value)

        elif operator == ">":
            lower_value = int(threshold) + 1
            current_lower = intervals[feature]["lower"]

            if current_lower is None:
                intervals[feature]["lower"] = lower_value
            else:
                intervals[feature]["lower"] = max(current_lower, lower_value)

    return intervals


def bin_range_for_feature(
    feature: str,
    lower: Optional[int],
    upper: Optional[int],
) -> Optional[str]:
    bins = NUMERIC_BINS.get(feature)

    if not bins:
        return None

    if lower is None:
        lower = -10**9

    if upper is None:
        upper = 10**9

    matching_labels: list[str] = []
    previous_max = -10**9

    for label, max_value in bins:
        bin_min = previous_max + 1
        bin_max = max_value

        intersects = not (upper < bin_min or lower > bin_max)

        if intersects:
            matching_labels.append(label)

        previous_max = max_value

    if not matching_labels:
        return None

    if len(matching_labels) == 1:
        return matching_labels[0]

    return "_OR_".join(matching_labels)


def compute_leaf_signature_dict(technical_rule: str) -> dict[str, str]:
    intervals = build_numeric_intervals(technical_rule)
    signature: dict[str, str] = {}

    for feature in FEATURE_ORDER:
        if feature not in intervals:
            continue

        lower = intervals[feature]["lower"]
        upper = intervals[feature]["upper"]
        bin_label = bin_range_for_feature(feature, lower, upper)

        if bin_label is not None:
            signature[feature] = bin_label

    return signature


def signature_to_string(signature: dict[str, str]) -> str:
    if not signature:
        return "ROOT"

    parts: list[str] = []

    for feature in FEATURE_ORDER:
        if feature in signature:
            parts.append(f"{feature}={signature[feature]}")

    extra_features = sorted(set(signature) - set(FEATURE_ORDER))

    for feature in extra_features:
        parts.append(f"{feature}={signature[feature]}")

    return " AND ".join(parts)


def signature_size(signature: dict[str, str]) -> int:
    return len(signature)


# ============================================================
# INTERPRETIVE LAYER
# ============================================================

def split_signature_value(value: str) -> set[str]:
    """
    Split binned values such as:
    - MODERATE_OR_CRITICAL
    - SMALL_OR_MEDIUM_OR_LARGE

    Important: split only on '_OR_', so labels like FAR_FROM_EDGE remain intact.
    """

    if value is None:
        return set()

    return set(str(value).split("_OR_"))


def signature_has(signature: dict[str, str], feature: str, expected) -> bool:
    """
    Check whether a binned signature contains a feature value.

    This supports combined bins such as MODERATE_OR_CRITICAL.
    """

    actual = signature.get(feature)

    if actual is None:
        return False

    actual_values = split_signature_value(actual)

    if isinstance(expected, (set, list, tuple)):
        expected_values = set(expected)
    else:
        expected_values = {expected}

    return bool(actual_values & expected_values)


def interpret_signature(signature: dict[str, str], predicted_phase: str) -> tuple[str, str]:
    """
    Optional human-readable layer.

    This does not define the primary grouping.
    The primary grouping is predicted_phase + binned_signature.
    """

    if not signature:
        return (
            "root_or_unrestricted_signature",
            "Το leaf δεν περιέχει αξιοποιήσιμα binned conditions. Η ερμηνεία βασίζεται κυρίως στην πρόβλεψη της φάσης.",
        )

    low_mobility = signature_has(
        signature,
        "bk_legal_moves_if_black_to_move",
        {"VERY_LOW", "LOW"},
    )

    edge_or_corner = (
        signature_has(signature, "bk_distance_to_edge", {"ON_EDGE", "NEAR_EDGE"})
        or signature_has(signature, "bk_distance_to_corner", {"AT_CORNER", "NEAR_CORNER"})
    )

    small_box = signature_has(signature, "queen_box_area", {"TINY", "SMALL"})
    large_box = signature_has(signature, "queen_box_area", "LARGE")

    high_queen_control = signature_has(
        signature,
        "queen_controls_bk_zone_count",
        "HIGH",
    )

    queen_close = signature_has(
        signature,
        "wq_bk_chebyshev_distance",
        {"CLOSE", "MEDIUM"},
    )

    queen_far = signature_has(
        signature,
        "wq_bk_chebyshev_distance",
        "FAR",
    )

    king_close = signature_has(
        signature,
        "wk_bk_chebyshev_distance",
        "CLOSE",
    ) or signature_has(
        signature,
        "wk_bk_manhattan_distance",
        "CLOSE",
    )

    king_far = signature_has(
        signature,
        "wk_bk_chebyshev_distance",
        "FAR",
    ) or signature_has(
        signature,
        "wk_bk_manhattan_distance",
        "FAR",
    )

    stalemate_moderate_or_critical = signature_has(
        signature,
        "stalemate_risk_level",
        {"MODERATE", "CRITICAL"},
    )

    stalemate_none = signature_has(
        signature,
        "stalemate_risk_level",
        "NONE",
    )

    if predicted_phase == "immediate_mate":
        if stalemate_moderate_or_critical:
            return (
                "immediate_stalemate_precision_signature",
                "Άμεση φάση: η θέση απαιτεί ακρίβεια λόγω πιθανού stalemate motif ή πολύ περιορισμένης κινητικότητας του μαύρου βασιλιά.",
            )

        if stalemate_none and queen_far:
            return (
                "immediate_quiet_queen_far_finish_signature",
                "Άμεσηη φάση χωρίς stalemate risk: ο λευκός βασιλιάς είναι κοντά, ενώ η βασίλισσα λειτουργεί από απόσταση για τον τελικό περιορισμό.",
            )

        if stalemate_none and queen_close:
            return (
                "immediate_quiet_queen_support_finish_signature",
                "Άμεση φάση χωρίς stalemate risk: η βασίλισσα βρίσκεται σε κοντινή ή μεσαία απόσταση και συμμετέχει άμεσα στο τελικό mating net.",
            )

        if signature_has(signature, "bk_distance_to_edge", "ON_EDGE") and king_close:
            return (
                "immediate_edge_king_support_signature",
                "Άμεση φάση: ο μαύρος βασιλιάς βρίσκεται στην άκρη και ο λευκός βασιλιάς είναι σε άμεση υποστηρικτική απόσταση.",
            )

        return (
            "immediate_other_data_signature",
            "Άμεση φάση που προκύπτει από data-driven signature χωρίς ειδικότερη ερμηνευτική ετικέτα.",
        )

    if predicted_phase == "short_conversion":
        if stalemate_moderate_or_critical and low_mobility:
            return (
                "short_stalemate_caution_signature",
                "Σύντομη μετατροπή: υπάρχει ένδειξη stalemate risk και περιορισμένη κινητικότητα του μαύρου βασιλιά, άρα απαιτείται προσεκτικός χειρισμός.",
            )

        if (
            signature_has(signature, "bk_distance_to_corner", {"AT_CORNER", "NEAR_CORNER"})
            and signature_has(signature, "bk_distance_to_edge", {"ON_EDGE", "NEAR_EDGE"})
        ):
            return (
                "short_corner_edge_pressure_signature",
                "Σύντομη μετατροπή: ο μαύρος βασιλιάς έχει οδηγηθεί σε άκρη ή κοντά σε γωνία, άρα η τελική πίεση έχει ήδη διαμορφωθεί.",
            )

        if small_box:
            return (
                "short_compact_queen_box_signature",
                "Σύντομη μετατροπή: το queen box είναι μικρό, οπότε ο διαθέσιμος χώρος του μαύρου βασιλιά έχει περιοριστεί σημαντικά.",
            )

        if (
            signature_has(signature, "bk_distance_to_edge", {"ON_EDGE", "NEAR_EDGE"})
            and signature_has(signature, "wk_bk_chebyshev_distance", {"CLOSE", "MEDIUM"})
        ):
            return (
                "short_edge_king_pressure_signature",
                "Σύντομη μετατροπή: ο μαύρος βασιλιάς βρίσκεται κοντά στην άκρη και ο λευκός βασιλιάς υποστηρίζει τη μετατροπή.",
            )

        if queen_close or high_queen_control:
            return (
                "short_queen_activity_signature",
                "Σύντομη μετατροπή: η βασίλισσα έχει ενεργό ρόλο είτε μέσω κοντινής/μεσαίας απόστασης είτε μέσω ελέγχου της ζώνης γύρω από τον μαύρο βασιλιά.",
            )

        return (
            "short_other_data_signature",
            "Σύντομη μετατροπή που περιγράφεται από data-driven signature χωρίς ειδικότερη ερμηνευτική ετικέτα.",
        )

    if predicted_phase == "medium_conversion":
        if small_box:
            return (
                "medium_queen_box_restriction_signature",
                "Μεσαία μετατροπή: η βασίλισσα έχει περιορίσει τον χώρο, αλλά η θέση δεν έχει φτάσει ακόμη στο τελικό mating net.",
            )

        if edge_or_corner:
            return (
                "medium_edge_progress_signature",
                "Μεσαία μετατροπή: ο μαύρος βασιλιάς έχει οδηγηθεί προς άκρη ή γωνία, αλλά χρειάζεται περαιτέρω τεχνική πρόοδος.",
            )

        if high_queen_control:
            return (
                "medium_queen_zone_control_signature",
                "Μεσαία μετατροπή: η βασίλισσα ελέγχει αρκετά τετράγωνα γύρω από τον μαύρο βασιλιά.",
            )

        if stalemate_moderate_or_critical:
            return (
                "medium_stalemate_caution_signature",
                "Μεσαία μετατροπή: υπάρχει ένδειξη stalemate risk, αλλά η θέση δεν ανήκει ακόμη σε άμεση τελική φάση.",
            )

        return (
            "medium_other_data_signature",
            "Μεσαία μετατροπή που προκύπτει από data-driven signature χωρίς ειδικότερη ερμηνευτική ετικέτα.",
        )

    if predicted_phase == "long_conversion":
        if large_box:
            return (
                "long_large_box_signature",
                "Μακρινή μετατροπή: το queen box παραμένει μεγάλο και ο μαύρος βασιλιάς έχει ακόμη αρκετό χώρο.",
            )

        if king_far:
            return (
                "long_white_king_far_signature",
                "Μακρινή μετατροπή: ο λευκός βασιλιάς βρίσκεται μακριά από τον μαύρο βασιλιά και δεν έχει ολοκληρωθεί ο συντονισμός.",
            )

        if signature_has(signature, "bk_distance_to_edge", "FAR_FROM_EDGE"):
            return (
                "long_black_king_central_signature",
                "Μακρινή μετατροπή: ο μαύρος βασιλιάς δεν έχει οδηγηθεί ακόμη στην άκρη.",
            )

        if stalemate_moderate_or_critical:
            return (
                "long_stalemate_diagnostic_signature",
                "Μακρινή μετατροπή με ένδειξη stalemate risk. Το μοτίβο είναι σπάνιο και λειτουργεί κυρίως διαγνωστικά.",
            )

        return (
            "long_other_data_signature",
            "Μακρινή μετατροπή που προκύπτει από data-driven signature χωρίς ειδικότερη ερμηνευτική ετικέτα.",
        )

    return (
        "unclassified_interpretive_signature",
        "Δεν υπάρχει ειδική ανθρώπινη ερμηνεία για αυτή τη φάση.",
    )


# ============================================================
# CONSOLIDATION PIPELINE
# ============================================================

def attach_signatures_to_leaves(all_leaves_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    for leaf_index, row in all_leaves_df.iterrows():
        technical_rule = str(row.get("technical_rule", "") or "")
        predicted_phase = str(row.get("predicted_phase", "") or "")

        signature_dict = compute_leaf_signature_dict(technical_rule)
        signature_string = signature_to_string(signature_dict)
        interpretive_label, interpretive_description = interpret_signature(
            signature=signature_dict,
            predicted_phase=predicted_phase,
        )

        samples = int(safe_to_numeric(row.get("samples"), default=0))
        purity = safe_to_numeric(row.get("purity"), default=0.0)

        rows.append({
            "leaf_index": leaf_index,
            "predicted_phase": predicted_phase,
            "binned_signature": signature_string,
            "signature_size": signature_size(signature_dict),
            "interpretive_label": interpretive_label,
            "interpretive_description": interpretive_description,
            "samples": samples,
            "purity": purity,
            "weighted_purity_mass": samples * purity,
            "technical_rule": technical_rule,
            "readable_rule": row.get("readable_rule", ""),
            "human_readable_rule": row.get("human_readable_rule", ""),
            "rule_summary": row.get("rule_summary", ""),
            "class_distribution": row.get("class_distribution", ""),
            "strategic_interpretation": row.get("strategic_interpretation", ""),
            "strategic_guidance": row.get("strategic_guidance", ""),
        })

    leaves = pd.DataFrame(rows)

    if leaves.empty:
        leaves["signature_id"] = pd.Series(dtype="object")
        return leaves

    leaves["phase_sort_order"] = leaves["predicted_phase"].apply(phase_sort_key)

    signature_keys = (
        leaves[["predicted_phase", "binned_signature"]]
        .drop_duplicates()
        .sort_values(["predicted_phase", "binned_signature"])
        .reset_index(drop=True)
    )

    signature_keys["signature_id"] = [
        f"KQK_SIG_{idx + 1:04d}" for idx in range(len(signature_keys))
    ]

    leaves = leaves.merge(
        signature_keys,
        on=["predicted_phase", "binned_signature"],
        how="left",
    )

    return leaves


def build_distinct_signatures(leaves: pd.DataFrame) -> pd.DataFrame:
    if leaves.empty:
        return pd.DataFrame()

    phase_totals = (
        leaves
        .groupby("predicted_phase")["samples"]
        .sum()
        .rename("phase_total_samples")
        .reset_index()
    )

    grouped = (
        leaves
        .groupby([
            "signature_id",
            "predicted_phase",
            "binned_signature",
            "signature_size",
        ])
        .agg(
            n_leaves=("leaf_index", "count"),
            total_samples=("samples", "sum"),
            weighted_purity_mass=("weighted_purity_mass", "sum"),
            mean_purity_unweighted=("purity", "mean"),
            max_leaf_samples=("samples", "max"),
        )
        .reset_index()
    )

    grouped = grouped.merge(
        phase_totals,
        on="predicted_phase",
        how="left",
    )

    grouped["coverage_within_phase"] = grouped["total_samples"] / grouped["phase_total_samples"]
    grouped["mean_purity_weighted"] = grouped["weighted_purity_mass"] / grouped["total_samples"]

    representative_rows = (
        leaves
        .sort_values(["signature_id", "samples", "purity"], ascending=[True, False, False])
        .groupby("signature_id", as_index=False)
        .head(1)
        [[
            "signature_id",
            "technical_rule",
            "human_readable_rule",
            "rule_summary",
            "interpretive_label",
            "interpretive_description",
        ]]
        .rename(columns={
            "technical_rule": "representative_full_path_rule",
            "human_readable_rule": "representative_human_readable_rule",
            "rule_summary": "representative_rule_summary",
        })
    )

    grouped = grouped.merge(
        representative_rows,
        on="signature_id",
        how="left",
    )

    grouped["phase_sort_order"] = grouped["predicted_phase"].apply(phase_sort_key)

    grouped = grouped.sort_values(
        [
            "phase_sort_order",
            "total_samples",
            "mean_purity_weighted",
            "n_leaves",
        ],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)

    grouped["rank_within_phase"] = (
        grouped
        .groupby("predicted_phase")["total_samples"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    grouped["objective_layer_note"] = (
        "Auto-extracted binned signature. The grouping is based only on "
        "predicted_phase and binned decision-tree conditions."
    )

    ordered_columns = [
        "signature_id",
        "predicted_phase",
        "rank_within_phase",
        "binned_signature",
        "signature_size",
        "n_leaves",
        "total_samples",
        "phase_total_samples",
        "coverage_within_phase",
        "mean_purity_weighted",
        "mean_purity_unweighted",
        "max_leaf_samples",
        "interpretive_label",
        "interpretive_description",
        "representative_full_path_rule",
        "representative_human_readable_rule",
        "representative_rule_summary",
        "objective_layer_note",
    ]

    existing_columns = [col for col in ordered_columns if col in grouped.columns]
    return grouped[existing_columns].copy()


def build_top_signatures_by_phase(
    distinct_signatures: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    if distinct_signatures.empty:
        return pd.DataFrame()

    return (
        distinct_signatures
        .sort_values(
            ["predicted_phase", "rank_within_phase"],
            ascending=[True, True],
        )
        .groupby("predicted_phase", group_keys=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def build_interpretive_motif_summary(leaves: pd.DataFrame) -> pd.DataFrame:
    if leaves.empty:
        return pd.DataFrame()

    phase_totals = (
        leaves
        .groupby("predicted_phase")["samples"]
        .sum()
        .rename("phase_total_samples")
        .reset_index()
    )

    grouped = (
        leaves
        .groupby([
            "predicted_phase",
            "interpretive_label",
            "interpretive_description",
        ])
        .agg(
            n_signatures=("signature_id", "nunique"),
            n_leaves=("leaf_index", "count"),
            total_samples=("samples", "sum"),
            weighted_purity_mass=("weighted_purity_mass", "sum"),
            mean_purity_unweighted=("purity", "mean"),
        )
        .reset_index()
    )

    grouped = grouped.merge(
        phase_totals,
        on="predicted_phase",
        how="left",
    )

    grouped["coverage_within_phase"] = grouped["total_samples"] / grouped["phase_total_samples"]
    grouped["mean_purity_weighted"] = grouped["weighted_purity_mass"] / grouped["total_samples"]
    grouped["phase_sort_order"] = grouped["predicted_phase"].apply(phase_sort_key)

    grouped = grouped.sort_values(
        ["phase_sort_order", "total_samples", "mean_purity_weighted"],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    return grouped.drop(columns=["weighted_purity_mass", "phase_sort_order"])


def build_phase_coverage(
    distinct_signatures: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    if distinct_signatures.empty:
        return pd.DataFrame()

    rows: list[dict] = []

    for phase, phase_df in distinct_signatures.groupby("predicted_phase"):
        phase_df = phase_df.sort_values("rank_within_phase")
        total_samples = float(phase_df["phase_total_samples"].iloc[0])

        top_df = phase_df.head(top_n)
        top_samples = float(top_df["total_samples"].sum())

        top_1_samples = float(phase_df.head(1)["total_samples"].sum())
        top_3_samples = float(phase_df.head(3)["total_samples"].sum())
        top_5_samples = float(phase_df.head(5)["total_samples"].sum())

        rows.append({
            "predicted_phase": phase,
            "n_distinct_signatures": len(phase_df),
            "phase_total_samples": int(total_samples),
            "top_1_coverage": top_1_samples / total_samples if total_samples else 0.0,
            "top_3_coverage": top_3_samples / total_samples if total_samples else 0.0,
            "top_5_coverage": top_5_samples / total_samples if total_samples else 0.0,
            f"top_{top_n}_coverage": top_samples / total_samples if total_samples else 0.0,
            f"top_{top_n}_samples": int(top_samples),
        })

    coverage = pd.DataFrame(rows)
    coverage["phase_sort_order"] = coverage["predicted_phase"].apply(phase_sort_key)

    return (
        coverage
        .sort_values("phase_sort_order")
        .drop(columns=["phase_sort_order"])
        .reset_index(drop=True)
    )


# ============================================================
# REPORT
# ============================================================

def format_percent(value: float) -> str:
    if pd.isna(value):
        return "0.00%"

    return f"{100 * float(value):.2f}%"


def write_markdown_report(
    output_path: Path,
    distinct_signatures: pd.DataFrame,
    top_signatures: pd.DataFrame,
    interpretive_motifs: pd.DataFrame,
    phase_coverage: pd.DataFrame,
    total_leaves: int,
    total_samples: int,
) -> None:
    lines: list[str] = []

    lines.extend([
        "# KQK Rule Consolidation — Auto-Extracted Signatures",
        "",
        "## 1. Σκοπός",
        "",
        "Το παρόν consolidation μετατρέπει τα leaves του decision tree σε "
        "**data-driven binned signatures**.",
        "",
        "Η κύρια ομαδοποίηση δεν βασίζεται σε χειροποίητα motifs. "
        "Κάθε signature προκύπτει αυτόματα από τα conditions του decision tree, "
        "μετά από μετατροπή των numeric thresholds σε κατηγορίες.",
        "",
        "Προαιρετικά προστίθεται ένα ανθρώπινο ερμηνευτικό label, αλλά αυτό "
        "δεν καθορίζει την κύρια ομαδοποίηση.",
        "",
        "## 2. Συνολικά στοιχεία",
        "",
        f"- Total leaves: **{total_leaves:,}**",
        f"- Total samples represented by leaves: **{total_samples:,}**",
        f"- Distinct auto-extracted signatures: **{len(distinct_signatures):,}**",
        "",
        "## 3. Coverage ανά phase",
        "",
    ])

    if phase_coverage.empty:
        lines.append("Δεν υπάρχουν διαθέσιμα στοιχεία coverage.")
    else:
        lines.append("| Phase | Distinct signatures | Samples | Top-1 | Top-3 | Top-5 | Top-N |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")

        for _, row in phase_coverage.iterrows():
            top_n_column = f"top_{TOP_N_SIGNATURES_PER_PHASE}_coverage"

            lines.append(
                "| "
                f"{row['predicted_phase']} | "
                f"{int(row['n_distinct_signatures'])} | "
                f"{int(row['phase_total_samples']):,} | "
                f"{format_percent(row['top_1_coverage'])} | "
                f"{format_percent(row['top_3_coverage'])} | "
                f"{format_percent(row['top_5_coverage'])} | "
                f"{format_percent(row[top_n_column])} |"
            )

    lines.extend([
        "",
        "## 4. Top auto-extracted signatures ανά phase",
        "",
    ])

    if top_signatures.empty:
        lines.append("Δεν υπάρχουν top signatures.")
    else:
        for phase in PHASE_ORDER:
            phase_df = top_signatures[top_signatures["predicted_phase"] == phase].copy()

            if phase_df.empty:
                continue

            lines.extend([
                f"### {PHASE_DISPLAY_NAMES.get(phase, phase)}",
                "",
                "| Rank | Signature ID | Coverage | Weighted purity | Signature | Interpretive label |",
                "|---:|---|---:|---:|---|---|",
            ])

            for _, row in phase_df.iterrows():
                signature = str(row["binned_signature"]).replace("|", "\\|")
                label = str(row.get("interpretive_label", "")).replace("|", "\\|")

                lines.append(
                    "| "
                    f"{int(row['rank_within_phase'])} | "
                    f"{row['signature_id']} | "
                    f"{format_percent(row['coverage_within_phase'])} | "
                    f"{float(row['mean_purity_weighted']):.4f} | "
                    f"`{signature}` | "
                    f"{label} |"
                )

            lines.append("")

    lines.extend([
        "",
        "## 5. Interpretive motif layer",
        "",
        "Το interpretive layer είναι βοηθητικό. Δεν παράγει την κύρια ομαδοποίηση. "
        "Χρησιμοποιείται μόνο για να δώσει σκακιστικά ονόματα σε signatures που "
        "προέκυψαν ήδη αυτόματα από τα δεδομένα.",
        "",
    ])

    if interpretive_motifs.empty:
        lines.append("Δεν υπάρχουν interpretive motifs.")
    else:
        lines.append("| Phase | Interpretive label | Signatures | Leaves | Coverage | Weighted purity |")
        lines.append("|---|---|---:|---:|---:|---:|")

        for _, row in interpretive_motifs.iterrows():
            lines.append(
                "| "
                f"{row['predicted_phase']} | "
                f"{row['interpretive_label']} | "
                f"{int(row['n_signatures'])} | "
                f"{int(row['n_leaves'])} | "
                f"{format_percent(row['coverage_within_phase'])} | "
                f"{float(row['mean_purity_weighted']):.4f} |"
            )

    lines.extend([
        "",
        "## 6. Μεθοδολογική σημείωση για τη διπλωματική",
        "",
        "Το consolidation αυτό είναι κατά βάση **αντικειμενικό / data-driven**, "
        "επειδή οι κύριες ομάδες προκύπτουν από distinct binned signatures και "
        "όχι από προκαθορισμένες σκακιστικές οικογένειες.",
        "",
        "Τα ερμηνευτικά labels είναι δεύτερο layer: βοηθούν στην παρουσίαση, "
        "αλλά δεν αλλάζουν το ποια leaves ανήκουν σε κάθε signature.",
        "",
        "Άρα η ανάλυση μπορεί να παρουσιαστεί ως:",
        "",
        "> automatic extraction of dominant binned rule signatures, followed by "
        "> optional human-readable chess interpretation.",
        "",
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Markdown report saved: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    CONSOLIDATION_DIR.mkdir(parents=True, exist_ok=True)

    print_section("KQK RULE CONSOLIDATION — AUTO-EXTRACTED SIGNATURES")
    print(f"Input:  {ALL_LEAF_RULES_CSV}")
    print(f"Output: {CONSOLIDATION_DIR}")

    if not ALL_LEAF_RULES_CSV.exists():
        print(
            f"\n[ERROR] Required input not found:\n"
            f"  {ALL_LEAF_RULES_CSV}\n\n"
            f"Run KQKDecTree.py first."
        )
        return

    all_leaves_df = read_csv_auto(ALL_LEAF_RULES_CSV)

    total_leaves = len(all_leaves_df)
    total_samples = int(
        pd.to_numeric(
            all_leaves_df.get("samples", pd.Series(dtype=float)),
            errors="coerce",
        )
        .fillna(0)
        .sum()
    )

    print(f"\nLoaded {total_leaves:,} leaf rules.")
    print(f"Total samples represented by leaves: {total_samples:,}")

    required_columns = {"predicted_phase", "technical_rule", "samples", "purity"}
    missing_columns = required_columns - set(all_leaves_df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns in all leaf rules CSV: {missing_columns}\n"
            f"Available columns: {all_leaves_df.columns.tolist()}"
        )

    print_section("BUILDING AUTO-EXTRACTED BINNED SIGNATURES")

    leaves_with_signature = attach_signatures_to_leaves(all_leaves_df)
    distinct_signatures = build_distinct_signatures(leaves_with_signature)
    top_signatures = build_top_signatures_by_phase(
        distinct_signatures=distinct_signatures,
        top_n=TOP_N_SIGNATURES_PER_PHASE,
    )
    interpretive_motifs = build_interpretive_motif_summary(leaves_with_signature)
    phase_coverage = build_phase_coverage(
        distinct_signatures=distinct_signatures,
        top_n=TOP_N_SIGNATURES_PER_PHASE,
    )

    print(f"\nDistinct auto-extracted signatures: {len(distinct_signatures):,}")

    if not phase_coverage.empty:
        print("\nPhase coverage:")
        display_cols = [
            "predicted_phase",
            "n_distinct_signatures",
            "phase_total_samples",
            "top_1_coverage",
            "top_3_coverage",
            "top_5_coverage",
            f"top_{TOP_N_SIGNATURES_PER_PHASE}_coverage",
        ]
        existing_display_cols = [
            col for col in display_cols
            if col in phase_coverage.columns
        ]
        print(phase_coverage[existing_display_cols].to_string(index=False))

    print_section("SAVING OUTPUTS")

    save_csv(
        leaves_with_signature,
        CONSOLIDATION_DIR / f"{OUTPUT_PREFIX}_leaves_with_signature.csv",
    )
    save_csv(
        distinct_signatures,
        CONSOLIDATION_DIR / f"{OUTPUT_PREFIX}_distinct_signatures.csv",
    )
    save_csv(
        top_signatures,
        CONSOLIDATION_DIR / f"{OUTPUT_PREFIX}_top_signatures_by_phase.csv",
    )
    save_csv(
        interpretive_motifs,
        CONSOLIDATION_DIR / f"{OUTPUT_PREFIX}_interpretive_motifs.csv",
    )
    save_csv(
        phase_coverage,
        CONSOLIDATION_DIR / f"{OUTPUT_PREFIX}_phase_coverage.csv",
    )

    write_markdown_report(
        output_path=CONSOLIDATION_DIR / f"{OUTPUT_PREFIX}_summary.md",
        distinct_signatures=distinct_signatures,
        top_signatures=top_signatures,
        interpretive_motifs=interpretive_motifs,
        phase_coverage=phase_coverage,
        total_leaves=total_leaves,
        total_samples=total_samples,
    )

    print_section("FILES SAVED")

    for path in sorted(CONSOLIDATION_DIR.glob(f"{OUTPUT_PREFIX}_*")):
        print(path)


if __name__ == "__main__":
    main()