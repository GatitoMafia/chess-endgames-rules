from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text


# ============================================================
# CONFIGURATION
# ============================================================

KRK_OUTPUT_ROOT = Path(r"C:\Users\Κωνσταντίνος\Desktop\KRK Dataset")

DATASET_DIR = KRK_OUTPUT_ROOT / "01_dataset"
MODEL_DIR = KRK_OUTPUT_ROOT / "02_model_dtz"

DATASET_CSV = DATASET_DIR / "krk_exhaustive_dtz_dataset.csv"

OUTPUT_PREFIX = "krk_dtz_decision_tree"
CSV_SEPARATOR = ";"
DECIMAL_PLACES = 4
FLOAT_FORMAT = f"%.{DECIMAL_PLACES}f"

pd.set_option("display.float_format", lambda value: f"{value:.{DECIMAL_PLACES}f}")

TARGET_COLUMN = "dtz_bucket"

PHASE_ORDER = [
    "long_conversion",
    "medium_conversion",
    "short_conversion",
    "immediate_mate_or_near",
]

PHASE_DISPLAY_NAMES = {
    "long_conversion": "Long conversion",
    "medium_conversion": "Medium conversion",
    "short_conversion": "Short conversion",
    "immediate_mate_or_near": "Immediate / near mate",
}

MAIN_SEED = 42
SEEDS = [42, 123, 456]
DEPTHS = [7, 9, 12]
CLASS_WEIGHT_OPTIONS = [None, "balanced"]

# Heuristic complexity penalty used to prefer simpler trees when performance is similar.
# It is used for model selection, not as a theoretical metric.
PENALTY_WEIGHT = 0.01

MIN_SAMPLES_SPLIT = 50
MIN_SAMPLES_LEAF = 25

FALLBACK_DEPTH = 12
FALLBACK_CLASS_WEIGHT: Optional[str] = "balanced"


# ============================================================
# FEATURE SET
# ============================================================

FEATURE_COLUMNS = [
    "bk_legal_moves_if_black_to_move",
    "bk_distance_to_corner",
    "bk_distance_to_edge",
    "bk_in_mating_zone",
    "wk_distance_to_edge",
    "wk_bk_chebyshev_distance",
    "wk_bk_manhattan_distance",
    "kings_in_direct_opposition",
    "wr_bk_file_distance",
    "wr_bk_rank_distance",
    "rook_cuts_off_black_king",
]

DIAGNOSTIC_COLUMNS = [
    "fen",
    "wdl",
    "dtz",
    "dtz_abs",
    "mate_moves",
    "dtz_bucket",
    *FEATURE_COLUMNS,
    "wk_file",
    "wk_rank",
    "wr_file",
    "wr_rank",
    "bk_file",
    "bk_rank",
]


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
        encoding="utf-8-sig",
    )

    if len(df.columns) == 1:
        df = pd.read_csv(path, encoding="utf-8-sig")

    return convert_numeric_like_columns(df)


def save_csv(df: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        path,
        index=index,
        encoding="utf-8-sig",
        sep=CSV_SEPARATOR,
        float_format=FLOAT_FORMAT,
        decimal=",",
    )


def validate_columns(df: pd.DataFrame) -> None:
    required_columns = [TARGET_COLUMN, *FEATURE_COLUMNS]
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
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


def phase_to_rank(phase: str) -> int:
    mapping = {phase: idx for idx, phase in enumerate(PHASE_ORDER)}
    return mapping.get(str(phase), -1)


def safe_int(value, default: int = -1) -> int:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return default
    return int(numeric)


# ============================================================
# STRATEGIC INTERPRETATION
# ============================================================

def get_phase_interpretation(phase: str) -> str:
    interpretations = {
        "long_conversion": (
            "Η θέση βρίσκεται ακόμα μακριά από την τελική μετατροπή. "
            "Ο μαύρος βασιλιάς έχει σχετικά αρκετό χώρο και ο περιορισμός δεν έχει ολοκληρωθεί."
        ),
        "medium_conversion": (
            "Η θέση βρίσκεται σε ενδιάμεσο στάδιο μετατροπής. "
            "Ο μαύρος βασιλιάς έχει αρχίσει να περιορίζεται, αλλά δεν υπάρχει ακόμα άμεσο mating net."
        ),
        "short_conversion": (
            "Η θέση είναι κοντά στην τελική μετατροπή. "
            "Ο μαύρος βασιλιάς είναι αρκετά περιορισμένος και η συνεργασία βασιλιά-πύργου "
            "αρχίζει να οδηγεί σε τελικό δίκτυο ματ."
        ),
        "immediate_mate_or_near": (
            "Η θέση βρίσκεται σε άμεσο ή σχεδόν άμεσο τελικό στάδιο. "
            "Ο μαύρος βασιλιάς είναι έντονα περιορισμένος και χρειάζεται συγκεκριμένη τελική ακολουθία."
        ),
    }
    return interpretations.get(phase, "Δεν υπάρχει διαθέσιμη ερμηνεία για αυτή τη φάση.")


def get_strategic_guidance(phase: str) -> str:
    guidance = {
        "long_conversion": (
            "Στρατηγικός στόχος: βελτίωση θέσης λευκού βασιλιά, σταδιακός περιορισμός "
            "του μαύρου βασιλιά και οδήγησή του προς την άκρη."
        ),
        "medium_conversion": (
            "Στρατηγικός στόχος: διατήρηση του περιορισμού και περαιτέρω μείωση "
            "του διαθέσιμου χώρου του μαύρου βασιλιά."
        ),
        "short_conversion": (
            "Στρατηγικός στόχος: διατήρηση του μαύρου βασιλιά στην άκρη ή σε περιορισμένη ζώνη "
            "και δημιουργία mating net."
        ),
        "immediate_mate_or_near": (
            "Στρατηγικός στόχος: αναζήτηση άμεσης τακτικής ή συγκεκριμένης τελικής ακολουθίας mate."
        ),
    }
    return guidance.get(phase, "Δεν υπάρχει διαθέσιμη στρατηγική οδηγία για αυτή τη φάση.")


# ============================================================
# RULE INTERPRETATION
# ============================================================

FRIENDLY_FEATURE_NAMES = {
    "bk_legal_moves_if_black_to_move": "black king legal moves",
    "bk_distance_to_corner": "black king distance to corner",
    "bk_distance_to_edge": "black king distance to edge",
    "bk_in_mating_zone": "black king in mating zone",
    "wk_distance_to_edge": "white king distance to edge",
    "wk_bk_chebyshev_distance": "white king - black king Chebyshev distance",
    "wk_bk_manhattan_distance": "white king - black king Manhattan distance",
    "kings_in_direct_opposition": "kings in direct opposition",
    "wr_bk_file_distance": "rook - black king file distance",
    "wr_bk_rank_distance": "rook - black king rank distance",
    "rook_cuts_off_black_king": "rook cuts off the black king",
}

BINARY_FEATURE_SENTENCES = {
    "rook_cuts_off_black_king": {
        1: "ο πύργος κόβει τον μαύρο βασιλιά δημιουργώντας γραμμή περιορισμού",
        0: "ο πύργος δεν δημιουργεί καθαρή γραμμή περιορισμού απέναντι στον μαύρο βασιλιά",
    },
    "bk_in_mating_zone": {
        1: "ο μαύρος βασιλιάς βρίσκεται στη ζώνη ματ",
        0: "ο μαύρος βασιλιάς δεν βρίσκεται ακόμα στη ζώνη ματ",
    },
    "kings_in_direct_opposition": {
        1: "οι βασιλιάδες βρίσκονται σε άμεση opposition",
        0: "οι βασιλιάδες δεν βρίσκονται σε άμεση opposition",
    },
}


def simplify_rule_text(rule: str) -> str:
    readable = rule
    for original, replacement in FRIENDLY_FEATURE_NAMES.items():
        readable = readable.replace(original, replacement)
    return readable


def parse_condition(condition: str):
    match = re.match(r"^\s*([A-Za-z0-9_]+)\s*(<=|>)\s*(-?\d+(?:\.\d+)?)\s*$", condition)
    if match is None:
        return None
    return match.group(1), match.group(2), float(match.group(3))


def format_integer_threshold(operator: str, threshold: float) -> tuple[str, int]:
    threshold_as_int = int(threshold)
    if operator == "<=":
        return "το πολύ", threshold_as_int
    return "τουλάχιστον", threshold_as_int + 1


def humanize_binary_condition(feature: str, operator: str, threshold: float) -> str | None:
    if feature not in BINARY_FEATURE_SENTENCES:
        return None
    if abs(threshold - 0.5) > 1e-9:
        return None
    feature_value = 0 if operator == "<=" else 1
    return BINARY_FEATURE_SENTENCES[feature][feature_value]


def humanize_numeric_condition(feature: str, operator: str, threshold: float) -> str:
    phrase, value = format_integer_threshold(operator, threshold)

    if feature == "bk_legal_moves_if_black_to_move":
        return f"ο μαύρος βασιλιάς έχει {phrase} {value} νόμιμες κινήσεις"
    if feature == "bk_distance_to_corner":
        return f"ο μαύρος βασιλιάς απέχει {phrase} {value} τετράγωνα από την κοντινότερη γωνία"
    if feature == "bk_distance_to_edge":
        return f"ο μαύρος βασιλιάς απέχει {phrase} {value} τετράγωνα από την άκρη"
    if feature == "wk_distance_to_edge":
        return f"ο λευκός βασιλιάς απέχει {phrase} {value} τετράγωνα από την άκρη"
    if feature == "wk_bk_chebyshev_distance":
        return f"οι δύο βασιλιάδες απέχουν {phrase} {value} τετράγωνα σε Chebyshev απόσταση"
    if feature == "wk_bk_manhattan_distance":
        return f"οι δύο βασιλιάδες απέχουν {phrase} {value} τετράγωνα σε Manhattan απόσταση"
    if feature == "wr_bk_file_distance":
        return f"ο πύργος απέχει {phrase} {value} κάθετες από τον μαύρο βασιλιά"
    if feature == "wr_bk_rank_distance":
        return f"ο πύργος απέχει {phrase} {value} οριζόντιες από τον μαύρο βασιλιά"

    readable_feature = FRIENDLY_FEATURE_NAMES.get(feature, feature)
    return f"{readable_feature} {operator} {threshold:.2f}"


def humanize_condition(condition: str) -> str:
    parsed = parse_condition(condition)
    if parsed is None:
        return simplify_rule_text(condition)

    feature, operator, threshold = parsed
    binary_sentence = humanize_binary_condition(feature, operator, threshold)
    if binary_sentence is not None:
        return binary_sentence

    return humanize_numeric_condition(feature, operator, threshold)


def build_human_readable_rule(technical_rule: str, predicted_phase: str) -> str:
    phase_name = PHASE_DISPLAY_NAMES.get(predicted_phase, predicted_phase)
    if technical_rule == "ROOT":
        return f"Δεν υπάρχουν επιπλέον συνθήκες. Η θέση ταξινομείται ως: {phase_name}."

    conditions = technical_rule.split(" AND ")
    human_conditions = [humanize_condition(condition) for condition in conditions]
    rule_body = " ΚΑΙ ".join(human_conditions)
    return f"Αν {rule_body}. Τότε η θέση ταξινομείται ως: {phase_name}."


def build_rule_summary(technical_rule: str, predicted_phase: str) -> str:
    low_mobility = (
        "bk_legal_moves_if_black_to_move <= 1.50" in technical_rule
        or "bk_legal_moves_if_black_to_move <= 2.50" in technical_rule
    )
    high_mobility = (
        "bk_legal_moves_if_black_to_move > 3.50" in technical_rule
        or "bk_legal_moves_if_black_to_move > 4.50" in technical_rule
    )
    black_close_to_edge = (
        "bk_distance_to_edge <= 0.50" in technical_rule
        or "bk_distance_to_edge <= 1.50" in technical_rule
    )
    black_far_from_edge = (
        "bk_distance_to_edge > 1.50" in technical_rule
        or "bk_distance_to_edge > 2.50" in technical_rule
    )
    black_close_to_corner = (
        "bk_distance_to_corner <= 0.50" in technical_rule
        or "bk_distance_to_corner <= 1.50" in technical_rule
    )
    kings_close = (
        "wk_bk_chebyshev_distance <= 1.50" in technical_rule
        or "wk_bk_chebyshev_distance <= 2.50" in technical_rule
    )
    rook_cuts_off = "rook_cuts_off_black_king > 0.50" in technical_rule
    bk_in_zone = "bk_in_mating_zone > 0.50" in technical_rule
    bk_not_in_zone = "bk_in_mating_zone <= 0.50" in technical_rule
    direct_opposition = "kings_in_direct_opposition > 0.50" in technical_rule

    if predicted_phase == "immediate_mate_or_near":
        if direct_opposition and bk_in_zone:
            return (
                "Immediate/near mate: οι βασιλιάδες βρίσκονται σε άμεση opposition "
                "με τον μαύρο στη ζώνη ματ. Η τελική ακολουθία είναι άμεση."
            )
        if direct_opposition and rook_cuts_off:
            return (
                "Immediate/near mate: άμεση opposition των βασιλιάδων μαζί με "
                "ενεργό κόψιμο του πύργου οδηγεί σε άμεσο mating net."
            )
        if black_close_to_corner and low_mobility:
            return (
                "Immediate/near mate: ο μαύρος βασιλιάς βρίσκεται πολύ κοντά στη γωνία "
                "και έχει έντονα περιορισμένη κινητικότητα."
            )
        if bk_in_zone and low_mobility:
            return (
                "Immediate/near mate: ο μαύρος βασιλιάς είναι καθηλωμένος στη ζώνη ματ "
                "με ελάχιστες νόμιμες κινήσεις."
            )
        if black_close_to_edge and low_mobility:
            return (
                "Immediate/near mate: ο μαύρος βασιλιάς είναι καθηλωμένος κοντά στην άκρη "
                "με ελάχιστες νόμιμες κινήσεις."
            )
        return "Immediate/near mate: η θέση αντιστοιχεί σε πολύ προχωρημένο στάδιο περιορισμού."

    if predicted_phase == "short_conversion":
        if bk_in_zone and rook_cuts_off:
            return "Short conversion: ο μαύρος βασιλιάς βρίσκεται στη ζώνη ματ και ο πύργος διατηρεί τη γραμμή περιορισμού."
        if black_close_to_edge:
            return "Short conversion: ο μαύρος βασιλιάς βρίσκεται κοντά στην άκρη και ο περιορισμός του είναι ήδη σημαντικός."
        if low_mobility:
            return "Short conversion: ο μαύρος βασιλιάς έχει περιορισμένη κινητικότητα, άρα η μετατροπή πλησιάζει."
        if rook_cuts_off:
            return "Short conversion: ο πύργος δημιουργεί γραμμή περιορισμού, βοηθώντας στο κόψιμο του μαύρου βασιλιά."
        return "Short conversion: η θέση είναι κοντά στη μετατροπή, αλλά δεν είναι ακόμα άμεσο mate."

    if predicted_phase == "medium_conversion":
        if bk_in_zone and not low_mobility:
            return "Medium conversion: ο μαύρος βασιλιάς βρίσκεται στη ζώνη ματ αλλά διατηρεί ακόμα κινητικότητα."
        if black_close_to_edge:
            return "Medium conversion: ο μαύρος βασιλιάς έχει αρχίσει να οδηγείται προς την άκρη, αλλά απαιτείται ακόμα τεχνική πρόοδος."
        if kings_close:
            return "Medium conversion: ο λευκός βασιλιάς έχει πλησιάσει και αρχίζει να υποστηρίζει τον περιορισμό."
        if rook_cuts_off:
            return "Medium conversion: ο πύργος συμβάλλει στον περιορισμό, αλλά η θέση δεν έχει φτάσει ακόμα σε άμεσο τελικό στάδιο."
        return "Medium conversion: η θέση δείχνει ενδιάμεσο στάδιο περιορισμού χωρίς άμεση τελική ακολουθία."

    if predicted_phase == "long_conversion":
        if bk_not_in_zone and high_mobility:
            return "Long conversion: ο μαύρος βασιλιάς δεν έχει οδηγηθεί στη ζώνη ματ και διατηρεί υψηλή κινητικότητα."
        if black_far_from_edge:
            return "Long conversion: ο μαύρος βασιλιάς έχει ακόμα αρκετό χώρο και δεν έχει περιοριστεί επαρκώς στην άκρη."
        if high_mobility:
            return "Long conversion: ο μαύρος βασιλιάς διατηρεί αρκετές νόμιμες κινήσεις, άρα η μετατροπή παραμένει μακριά."
        return "Long conversion: η θέση απαιτεί ακόμα σημαντική βελτίωση και περιορισμό."

    return "Δεν υπάρχει διαθέσιμη σύντομη στρατηγική περίληψη για αυτή τη φάση."


# ============================================================
# ERROR ANALYSIS
# ============================================================

def classify_error_row(row: pd.Series) -> tuple[str, str, str]:
    true_phase = str(row["true"])
    pred_phase = str(row["pred"])
    true_rank = phase_to_rank(true_phase)
    pred_rank = phase_to_rank(pred_phase)
    flags: list[str] = []

    if true_rank == -1 or pred_rank == -1:
        return "unknown_phase_confusion", "unknown_phase_confusion", "unknown_phase_confusion"

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
    elif pair == {"short_conversion", "immediate_mate_or_near"}:
        flags.append("short_immediate_boundary_confusion")
    elif pair == {"long_conversion", "short_conversion"}:
        flags.append("long_short_large_confusion")
    elif pair == {"medium_conversion", "immediate_mate_or_near"}:
        flags.append("medium_immediate_large_confusion")
    elif pair == {"long_conversion", "immediate_mate_or_near"}:
        flags.append("long_immediate_extreme_confusion")

    bk_legal_moves = safe_int(row.get("bk_legal_moves_if_black_to_move", -1))
    bk_distance_to_edge = safe_int(row.get("bk_distance_to_edge", -1))
    bk_distance_to_corner = safe_int(row.get("bk_distance_to_corner", -1))
    bk_in_zone = safe_int(row.get("bk_in_mating_zone", 0), default=0)
    wk_bk_chebyshev = safe_int(row.get("wk_bk_chebyshev_distance", -1))
    wk_bk_manhattan = safe_int(row.get("wk_bk_manhattan_distance", -1))
    wk_distance_to_edge = safe_int(row.get("wk_distance_to_edge", -1))
    direct_opposition = safe_int(row.get("kings_in_direct_opposition", 0), default=0)
    wr_bk_file_distance = safe_int(row.get("wr_bk_file_distance", -1))
    wr_bk_rank_distance = safe_int(row.get("wr_bk_rank_distance", -1))
    rook_cuts_off = safe_int(row.get("rook_cuts_off_black_king", 0), default=0)

    if phase_distance > 0:
        if bk_legal_moves >= 4:
            flags.append("black_king_freedom_overestimated_as_restriction")
        if bk_distance_to_edge >= 2:
            flags.append("edge_distance_overestimated_as_restriction")
        if bk_distance_to_corner >= 3:
            flags.append("corner_distance_overestimated_as_restriction")
        if bk_in_zone == 0:
            flags.append("mating_zone_status_overestimated")
        if wk_bk_chebyshev >= 4 or wk_bk_manhattan >= 7:
            flags.append("white_king_distance_underestimated")
        if wk_distance_to_edge >= 3:
            flags.append("white_king_edge_position_misread")
        if wr_bk_file_distance >= 4 and wr_bk_rank_distance >= 4:
            flags.append("rook_distance_underestimated")

    if phase_distance < 0:
        if 0 <= bk_legal_moves <= 2:
            flags.append("black_king_restriction_underestimated")
        if bk_distance_to_edge <= 0:
            flags.append("edge_restriction_underestimated")
        if bk_distance_to_corner <= 0:
            flags.append("corner_restriction_underestimated")
        if bk_in_zone == 1:
            flags.append("mating_zone_status_underestimated")
        if direct_opposition == 1:
            flags.append("direct_opposition_significance_underestimated")
        if 0 <= wk_bk_chebyshev <= 2 or 0 <= wk_bk_manhattan <= 3:
            flags.append("white_king_support_underestimated")
        if wr_bk_file_distance <= 1 or wr_bk_rank_distance <= 1:
            flags.append("rook_proximity_underestimated")

    if rook_cuts_off:
        flags.append("rook_cutoff_related_confusion")

    priority_order = [
        "long_immediate_extreme_confusion",
        "medium_immediate_large_confusion",
        "long_short_large_confusion",
        "short_immediate_boundary_confusion",
        "medium_short_boundary_confusion",
        "long_medium_boundary_confusion",
        "direct_opposition_significance_underestimated",
        "mating_zone_status_underestimated",
        "mating_zone_status_overestimated",
        "black_king_freedom_overestimated_as_restriction",
        "edge_distance_overestimated_as_restriction",
        "corner_distance_overestimated_as_restriction",
        "white_king_distance_underestimated",
        "white_king_edge_position_misread",
        "rook_distance_underestimated",
        "black_king_restriction_underestimated",
        "edge_restriction_underestimated",
        "corner_restriction_underestimated",
        "white_king_support_underestimated",
        "rook_proximity_underestimated",
        "rook_cutoff_related_confusion",
        "large_phase_confusion",
        "predicted_too_close_to_mate",
        "predicted_too_far_from_mate",
    ]

    primary_category = next((candidate for candidate in priority_order if candidate in flags), error_type)
    all_categories = " | ".join(dict.fromkeys(flags))
    return error_type, primary_category, all_categories


def categorize_errors(errors_df: pd.DataFrame) -> pd.DataFrame:
    categorized = errors_df.copy()

    if categorized.empty:
        for col in ["error_type", "primary_error_category", "all_error_flags"]:
            categorized[col] = pd.Series(dtype="object")
        for col in ["phase_distance", "true_rank", "pred_rank", "absolute_phase_distance"]:
            categorized[col] = pd.Series(dtype="float")
        return categorized

    triples = categorized.apply(classify_error_row, axis=1)
    categorized["error_type"] = [triple[0] for triple in triples]
    categorized["primary_error_category"] = [triple[1] for triple in triples]
    categorized["all_error_flags"] = [triple[2] for triple in triples]
    categorized["true_rank"] = categorized["true"].apply(phase_to_rank)
    categorized["pred_rank"] = categorized["pred"].apply(phase_to_rank)
    categorized["phase_distance"] = categorized["pred_rank"] - categorized["true_rank"]
    categorized["absolute_phase_distance"] = categorized["phase_distance"].abs()

    return categorized


def explode_error_flags(categorized_errors: pd.DataFrame) -> pd.DataFrame:
    if categorized_errors.empty or "all_error_flags" not in categorized_errors.columns:
        return pd.DataFrame(columns=["error_flag", "count"])

    rows: list[dict] = []
    for _, row in categorized_errors.iterrows():
        flags = str(row["all_error_flags"]).split(" | ")
        for flag in flags:
            rows.append({
                "error_flag": flag,
                "true": row.get("true"),
                "pred": row.get("pred"),
                "error_type": row.get("error_type"),
                "phase_distance": row.get("phase_distance"),
                "dtz_abs": row.get("dtz_abs"),
                "mate_moves": row.get("mate_moves"),
            })

    return pd.DataFrame(rows)


# ============================================================
# MODEL FUNCTIONS
# ============================================================

def make_classifier(max_depth: int, class_weight: Optional[str], seed: int) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(
        criterion="gini",
        max_depth=max_depth,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        class_weight=class_weight,
        random_state=seed,
    )


def add_diagnostics(results: pd.DataFrame, df: pd.DataFrame, indices: pd.Index) -> pd.DataFrame:
    enriched = results.copy()
    for col in DIAGNOSTIC_COLUMNS:
        if col in df.columns and col not in enriched.columns:
            enriched[col] = df.loc[indices, col].values
    return enriched


def train_and_evaluate(
    df: pd.DataFrame,
    *,
    max_depth: int,
    class_weight: Optional[str],
    seed: int,
) -> dict:
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=seed,
        stratify=y,
    )

    clf = make_classifier(max_depth=max_depth, class_weight=class_weight, seed=seed)
    clf.fit(X_train, y_train)

    y_train_pred = clf.predict(X_train)
    y_pred = clf.predict(X_test)

    cm = confusion_matrix(y_test, y_pred, labels=PHASE_ORDER)

    results = X_test.copy()
    results["true"] = y_test.values
    results["pred"] = y_pred
    results = add_diagnostics(results, df, X_test.index)

    errors = results[results["true"] != results["pred"]].copy()
    categorized_errors = categorize_errors(errors)

    importances = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance": clf.feature_importances_,
    }).sort_values("importance", ascending=False)

    return {
        "model": clf,
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
        "confusion_matrix": cm,
        "total_errors": len(errors),
        "errors": errors,
        "categorized_errors": categorized_errors,
        "importances": importances,
        "classification_report": classification_report(
            y_test,
            y_pred,
            labels=PHASE_ORDER,
            target_names=[PHASE_DISPLAY_NAMES[p] for p in PHASE_ORDER],
            zero_division=0,
            digits=DECIMAL_PLACES,
        ),
        "rules_text": export_text(clf, feature_names=FEATURE_COLUMNS, decimals=DECIMAL_PLACES),
    }


def cross_validate_main_model(
    df: pd.DataFrame,
    *,
    max_depth: int,
    class_weight: Optional[str],
    seed: int,
) -> dict[str, float]:
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    clf = make_classifier(max_depth=max_depth, class_weight=class_weight, seed=seed)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    acc = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    balanced_acc = cross_val_score(clf, X, y, cv=cv, scoring="balanced_accuracy")
    macro_f1 = cross_val_score(clf, X, y, cv=cv, scoring="f1_macro")

    return {
        "cv_accuracy_mean": float(acc.mean()),
        "cv_accuracy_std": float(acc.std()),
        "cv_balanced_accuracy_mean": float(balanced_acc.mean()),
        "cv_balanced_accuracy_std": float(balanced_acc.std()),
        "cv_macro_f1_mean": float(macro_f1.mean()),
        "cv_macro_f1_std": float(macro_f1.std()),
    }


# ============================================================
# RULE EXTRACTION
# ============================================================

def extract_leaf_rules(clf: DecisionTreeClassifier) -> pd.DataFrame:
    tree = clf.tree_
    classes = clf.classes_
    rows: list[dict] = []

    def recurse(node_id: int, conditions: list[str]) -> None:
        left = tree.children_left[node_id]
        right = tree.children_right[node_id]

        if left == right:
            values = tree.value[node_id][0]
            total = int(tree.n_node_samples[node_id])

            if total == 0 or values.sum() == 0:
                return

            predicted_index = int(values.argmax())
            predicted_phase = str(classes[predicted_index])
            purity = float(values[predicted_index] / values.sum())
            technical_rule = " AND ".join(conditions) if conditions else "ROOT"
            readable_rule = simplify_rule_text(technical_rule)

            rows.append({
                "predicted_phase": predicted_phase,
                "samples": total,
                "purity": round(purity, DECIMAL_PLACES),
                "technical_rule": technical_rule,
                "readable_rule": readable_rule,
                "human_readable_rule": build_human_readable_rule(technical_rule, predicted_phase),
                "rule_summary": build_rule_summary(technical_rule, predicted_phase),
                "class_distribution": {
                    str(classes[i]): round(float(values[i]), DECIMAL_PLACES)
                    for i in range(len(classes))
                },
                "strategic_interpretation": get_phase_interpretation(predicted_phase),
                "strategic_guidance": get_strategic_guidance(predicted_phase),
            })
            return

        feature = FEATURE_COLUMNS[tree.feature[node_id]]
        threshold = tree.threshold[node_id]
        recurse(left, conditions + [f"{feature} <= {threshold:.2f}"])
        recurse(right, conditions + [f"{feature} > {threshold:.2f}"])

    recurse(0, [])
    rules = pd.DataFrame(rows)

    if rules.empty:
        return rules

    rules["phase_rank"] = rules["predicted_phase"].apply(phase_to_rank)
    rules = rules.sort_values(
        by=["phase_rank", "samples", "purity"],
        ascending=[True, False, False],
    )

    return rules.drop(columns=["phase_rank"])


# ============================================================
# EXPORT HELPERS
# ============================================================

def build_categorized_errors_export(categorized_errors: pd.DataFrame) -> pd.DataFrame:
    if categorized_errors.empty:
        return categorized_errors

    preferred_columns = [
        "fen",
        "true",
        "pred",
        "true_rank",
        "pred_rank",
        "phase_distance",
        "error_type",
        "primary_error_category",
        "all_error_flags",
        "dtz",
        "dtz_abs",
        "mate_moves",
        "dtz_bucket",
    ]
    existing_columns = [col for col in preferred_columns if col in categorized_errors.columns]
    return categorized_errors[existing_columns].copy()


def build_error_summaries(categorized_errors: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if categorized_errors.empty:
        return {
            "error_category_counts": pd.DataFrame(),
            "error_type_counts": pd.DataFrame(),
            "phase_confusion_counts": pd.DataFrame(),
            "errors_by_dtz_abs": pd.DataFrame(),
            "errors_by_mate_moves": pd.DataFrame(),
            "error_flag_counts": pd.DataFrame(),
            "error_severity_counts": pd.DataFrame(),
        }

    error_category_counts = (
        categorized_errors["primary_error_category"]
        .value_counts()
        .rename_axis("primary_error_category")
        .reset_index(name="count")
    )

    error_type_counts = (
        categorized_errors["error_type"]
        .value_counts()
        .rename_axis("error_type")
        .reset_index(name="count")
    )

    phase_confusion_counts = (
        categorized_errors
        .groupby(["true", "pred"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    errors_by_dtz_abs = (
        categorized_errors
        .groupby(["dtz_abs", "error_type"])
        .size()
        .reset_index(name="count")
        .sort_values(["dtz_abs", "error_type"])
        if "dtz_abs" in categorized_errors.columns
        else pd.DataFrame()
    )

    errors_by_mate_moves = (
        categorized_errors
        .groupby(["mate_moves", "error_type"])
        .size()
        .reset_index(name="count")
        .sort_values(["mate_moves", "error_type"])
        if "mate_moves" in categorized_errors.columns
        else pd.DataFrame()
    )

    exploded_flags = explode_error_flags(categorized_errors)
    error_flag_counts = (
        exploded_flags["error_flag"]
        .value_counts()
        .rename_axis("error_flag")
        .reset_index(name="count")
        if not exploded_flags.empty
        else pd.DataFrame()
    )

    error_severity_counts = (
        categorized_errors["absolute_phase_distance"]
        .value_counts()
        .rename_axis("absolute_phase_distance")
        .reset_index(name="count")
        .sort_values("absolute_phase_distance")
    )

    return {
        "error_category_counts": error_category_counts,
        "error_type_counts": error_type_counts,
        "phase_confusion_counts": phase_confusion_counts,
        "errors_by_dtz_abs": errors_by_dtz_abs,
        "errors_by_mate_moves": errors_by_mate_moves,
        "error_flag_counts": error_flag_counts,
        "error_severity_counts": error_severity_counts,
    }


def run_multi_seed_stability(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for seed in SEEDS:
        for depth in DEPTHS:
            for class_weight in CLASS_WEIGHT_OPTIONS:
                result = train_and_evaluate(df, max_depth=depth, class_weight=class_weight, seed=seed)
                rows.append({
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
                })
                print(
                    f"seed={seed} | depth={depth} | class_weight={class_weight} | "
                    f"acc={result['test_accuracy']:.4f} | "
                    f"bal_acc={result['balanced_accuracy']:.4f} | "
                    f"macro_f1={result['macro_f1']:.4f} | "
                    f"errors={result['total_errors']}"
                )

    return pd.DataFrame(rows)


def summarize_stability(raw_df: pd.DataFrame) -> pd.DataFrame:
    return (
        raw_df.groupby(["max_depth", "class_weight"])
        .agg(
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
        )
        .reset_index()
    )


# ============================================================
# AUTOMATIC MAIN-MODEL SELECTION
# ============================================================

def compute_depth_selection_scores(
    stability_summary: pd.DataFrame,
    penalty_weight: float = PENALTY_WEIGHT,
) -> pd.DataFrame:
    if stability_summary.empty:
        return pd.DataFrame()

    scored = stability_summary.copy()
    scored["class_weight_normalized"] = scored["class_weight"].apply(normalize_class_weight_value)
    max_leaves = scored["number_of_leaves_mean"].max()

    if max_leaves is None or max_leaves <= 0:
        return pd.DataFrame()

    scored["complexity_penalty"] = penalty_weight * np.sqrt(scored["number_of_leaves_mean"] / max_leaves)
    scored["adjusted_score"] = scored["balanced_accuracy_mean"] - scored["complexity_penalty"]
    return scored.sort_values("adjusted_score", ascending=False).reset_index(drop=True)


def select_optimal_main_model(
    stability_summary: pd.DataFrame,
    penalty_weight: float = PENALTY_WEIGHT,
) -> dict:
    scored = compute_depth_selection_scores(stability_summary, penalty_weight)

    if scored.empty:
        return {
            "optimal_depth": None,
            "optimal_class_weight_normalized": None,
            "message": "No stability rows available — selection skipped.",
            "all_scores": pd.DataFrame(),
        }

    best_row = scored.iloc[0]
    selection_columns = [
        "max_depth",
        "class_weight",
        "class_weight_normalized",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
        "macro_f1_mean",
        "macro_f1_std",
        "total_errors_mean",
        "total_errors_std",
        "number_of_leaves_mean",
        "number_of_leaves_std",
        "complexity_penalty",
        "adjusted_score",
    ]

    return {
        "optimal_depth": int(best_row["max_depth"]),
        "optimal_class_weight_normalized": str(best_row["class_weight_normalized"]),
        "balanced_accuracy": float(best_row["balanced_accuracy_mean"]),
        "macro_f1": float(best_row["macro_f1_mean"]),
        "total_errors": float(best_row["total_errors_mean"]),
        "n_leaves": float(best_row["number_of_leaves_mean"]),
        "complexity_penalty": float(best_row["complexity_penalty"]),
        "adjusted_score": float(best_row["adjusted_score"]),
        "all_scores": scored[selection_columns].copy(),
    }


def save_depth_selection_outputs(model_dir: Path, selection_info: dict) -> None:
    if selection_info["optimal_depth"] is None:
        print("\nDepth selection skipped:")
        print(selection_info.get("message", "Unknown reason."))
        return

    save_csv(selection_info["all_scores"], model_dir / f"{OUTPUT_PREFIX}_depth_selection_scores.csv")

    selected_depth_summary = pd.DataFrame([{
        "optimal_depth": selection_info["optimal_depth"],
        "optimal_class_weight": selection_info["optimal_class_weight_normalized"],
        "balanced_accuracy_mean": selection_info["balanced_accuracy"],
        "macro_f1_mean": selection_info["macro_f1"],
        "total_errors_mean": selection_info["total_errors"],
        "total_errors_mean_rounded": int(round(selection_info["total_errors"])),
        "n_leaves_mean": selection_info["n_leaves"],
        "n_leaves_mean_rounded": int(round(selection_info["n_leaves"])),
        "complexity_penalty": selection_info["complexity_penalty"],
        "adjusted_score": selection_info["adjusted_score"],
        "penalty_weight": PENALTY_WEIGHT,
    }])

    save_csv(selected_depth_summary, model_dir / f"{OUTPUT_PREFIX}_selected_optimal_depth.csv")
    print("\nDepth selection diagnostic (best of the full grid):")
    print(selected_depth_summary.to_string(index=False))


# ============================================================
# SUMMARY REPORT
# ============================================================

def write_summary_report(
    model_dir: Path,
    main_result: dict,
    cv: dict[str, float],
    stability_summary: pd.DataFrame,
    important_rules: pd.DataFrame,
    selected_main_depth: int,
    selected_main_class_weight: Optional[str],
    selection_info: dict,
) -> None:
    overfitting_gap = main_result["train_accuracy"] - main_result["test_accuracy"]

    if overfitting_gap > 0.05:
        overfitting_comment = "Πιθανή ένδειξη overfitting: η διαφορά train-test είναι σχετικά μεγάλη."
    else:
        overfitting_comment = "Δεν υπάρχει έντονη ένδειξη overfitting: η διαφορά train-test είναι μικρή."

    selected_class_weight_normalized = normalize_class_weight_value(selected_main_class_weight)
    selected_stability = stability_summary[
        (stability_summary["max_depth"] == selected_main_depth)
        & (stability_summary["class_weight"].apply(normalize_class_weight_value) == selected_class_weight_normalized)
    ].copy()

    top_importances_text = main_result["importances"].head(10).to_string(index=False)
    selected_stability_text = (
        selected_stability.to_string(index=False)
        if not selected_stability.empty
        else "Δεν υπάρχουν διαθέσιμα αποτελέσματα σταθερότητας για το επιλεγμένο συνδυασμό."
    )

    if important_rules.empty:
        representative_rules_text = "Δεν εξήχθησαν σημαντικοί/αντιπροσωπευτικοί κανόνες."
    else:
        rule_preview_columns = ["predicted_phase", "samples", "purity", "human_readable_rule", "rule_summary"]
        existing_columns = [col for col in rule_preview_columns if col in important_rules.columns]
        representative_rules_text = important_rules[existing_columns].head(8).to_string(index=False)

    report_lines = [
        "# KRK DTZ Decision Tree — Summary Report",
        "",
        "## 1. Main Model Configuration",
        "",
        "| Parameter | Value |",
        "|---|---:|",
        f"| Seed | {MAIN_SEED} |",
        f"| Max depth (auto-selected) | {selected_main_depth} |",
        f"| Class weight (auto-selected) | {selected_main_class_weight} |",
        f"| Min samples split | {MIN_SAMPLES_SPLIT} |",
        f"| Min samples leaf | {MIN_SAMPLES_LEAF} |",
        f"| Penalty weight | {PENALTY_WEIGHT} |",
        f"| Target column | {TARGET_COLUMN} |",
        "",
        "## 2. Selection Diagnostic",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| Adjusted score | {selection_info['adjusted_score']:.4f} |",
        f"| Balanced accuracy (multi-seed mean) | {selection_info['balanced_accuracy']:.4f} |",
        f"| Macro F1 (multi-seed mean) | {selection_info['macro_f1']:.4f} |",
        f"| Complexity penalty | {selection_info['complexity_penalty']:.4f} |",
        f"| Mean leaves at selected config | {selection_info['n_leaves']:.2f} |",
        "",
        "## 3. Target Phases",
        "",
        "Ο στόχος πρόβλεψης είναι η στήλη `dtz_bucket`, με τις παρακάτω φάσεις μετατροπής:",
        "",
        str(PHASE_ORDER),
        "",
        "## 4. Final Model Performance Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Train accuracy | {main_result['train_accuracy']:.4f} |",
        f"| Test accuracy | {main_result['test_accuracy']:.4f} |",
        f"| Accuracy gap | {overfitting_gap:.4f} |",
        f"| Balanced accuracy | {main_result['balanced_accuracy']:.4f} |",
        f"| Macro F1 | {main_result['macro_f1']:.4f} |",
        f"| Weighted F1 | {main_result['weighted_f1']:.4f} |",
        f"| CV accuracy | {cv['cv_accuracy_mean']:.4f} ± {cv['cv_accuracy_std']:.4f} |",
        f"| CV balanced accuracy | {cv['cv_balanced_accuracy_mean']:.4f} ± {cv['cv_balanced_accuracy_std']:.4f} |",
        f"| CV macro F1 | {cv['cv_macro_f1_mean']:.4f} ± {cv['cv_macro_f1_std']:.4f} |",
        "",
        f"**Σχόλιο για overfitting:** {overfitting_comment}",
        "",
        "## 5. Tree Complexity",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| Actual depth | {main_result['actual_depth']} |",
        f"| Number of leaves | {main_result['number_of_leaves']} |",
        f"| Total errors | {main_result['total_errors']} |",
        "",
        "## 6. Top 10 Feature Importances",
        "",
        top_importances_text,
        "",
        "## 7. Multi-Seed Stability for Selected (Depth, ClassWeight)",
        "",
        selected_stability_text,
        "",
        "## 8. Representative Extracted Rules",
        "",
        representative_rules_text,
        "",
        "Τα πλήρη αρχεία κανόνων αποθηκεύονται στα:",
        "",
        f"- `{OUTPUT_PREFIX}_all_leaf_rules.csv`",
        f"- `{OUTPUT_PREFIX}_important_rules.csv`",
        "",
        "## 9. Interpretation Note",
        "",
        "Οι εξαγόμενοι κανόνες προκύπτουν από decision tree που εκπαιδεύτηκε σε KRK θέσεις επισημασμένες από Syzygy tablebases.",
        "",
        "Πρόκειται για machine-learned approximations σκακιστικών εννοιών και όχι για τυπικά σκακιστικά θεωρήματα.",
    ]

    report_path = model_dir / f"{OUTPUT_PREFIX}_summary_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("\nΑποθηκεύτηκε το summary report στο:")
    print(report_path)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("KRKDecTree started.")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print_section("KRK FINAL DTZ DECISION TREE ANALYSIS")
    print(f"Dataset: {DATASET_CSV}")
    print(f"Output folder: {MODEL_DIR}")

    df = read_csv_auto(DATASET_CSV)
    validate_columns(df)

    print(f"\nRows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\nTarget distribution:")
    print(df[TARGET_COLUMN].value_counts())

    print("\nTarget distribution (%):")
    print((df[TARGET_COLUMN].value_counts(normalize=True) * 100).round(DECIMAL_PLACES))

    if "dtz_abs" in df.columns:
        print("\nDTZ statistics:")
        print(df["dtz_abs"].describe())

    if "mate_moves" in df.columns:
        print("\nMate-move statistics:")
        print(df["mate_moves"].describe())

    print_section("FEATURE SET USED")
    for feature in FEATURE_COLUMNS:
        print(f"- {feature}")

    print_section("MULTI-SEED STABILITY")
    multi_seed_raw = run_multi_seed_stability(df)
    stability_summary = summarize_stability(multi_seed_raw)

    save_csv(multi_seed_raw, MODEL_DIR / f"{OUTPUT_PREFIX}_multi_seed_raw_results.csv")
    save_csv(stability_summary, MODEL_DIR / f"{OUTPUT_PREFIX}_multi_seed_stability_summary.csv")

    print_section("AUTOMATIC MAIN-MODEL SELECTION")
    selection_info = select_optimal_main_model(
        stability_summary=stability_summary,
        penalty_weight=PENALTY_WEIGHT,
    )

    save_depth_selection_outputs(
        model_dir=MODEL_DIR,
        selection_info=selection_info,
    )

    if selection_info["optimal_depth"] is None:
        selected_main_depth = FALLBACK_DEPTH
        selected_main_class_weight = FALLBACK_CLASS_WEIGHT
        print(
            f"\nNo optimal configuration selected. "
            f"Falling back to depth={FALLBACK_DEPTH}, class_weight={FALLBACK_CLASS_WEIGHT}."
        )
    else:
        selected_main_depth = selection_info["optimal_depth"]
        selected_main_class_weight = parse_class_weight_for_sklearn(
            selection_info["optimal_class_weight_normalized"]
        )
        print("\nUsing automatically selected main-model configuration:")
        print(f"  - max_depth = {selected_main_depth}")
        print(f"  - class_weight = {selected_main_class_weight}")
        print(f"  - adjusted_score = {selection_info['adjusted_score']:.4f}")

    print_section("MAIN MODEL (TRAINED ON AUTO-SELECTED CONFIGURATION)")
    main_result = train_and_evaluate(
        df,
        max_depth=selected_main_depth,
        class_weight=selected_main_class_weight,
        seed=MAIN_SEED,
    )

    cv = cross_validate_main_model(
        df,
        max_depth=selected_main_depth,
        class_weight=selected_main_class_weight,
        seed=MAIN_SEED,
    )

    main_summary = pd.DataFrame([{
        "seed": MAIN_SEED,
        "max_depth": selected_main_depth,
        "class_weight": str(selected_main_class_weight),
        "selected_via": "adjusted_score (balanced_acc - complexity_penalty)",
        "penalty_weight": PENALTY_WEIGHT,
        "adjusted_score_at_selection": selection_info["adjusted_score"],
        "train_accuracy": main_result["train_accuracy"],
        "test_accuracy": main_result["test_accuracy"],
        "accuracy_gap": main_result["train_accuracy"] - main_result["test_accuracy"],
        "balanced_accuracy": main_result["balanced_accuracy"],
        "macro_f1": main_result["macro_f1"],
        "weighted_f1": main_result["weighted_f1"],
        **cv,
        "actual_depth": main_result["actual_depth"],
        "number_of_leaves": main_result["number_of_leaves"],
        "total_errors": main_result["total_errors"],
    }])

    print(main_summary.to_string(index=False))

    print("\nClassification report:")
    print(main_result["classification_report"])

    confusion_df = pd.DataFrame(
        main_result["confusion_matrix"],
        index=[f"true_{phase}" for phase in PHASE_ORDER],
        columns=[f"pred_{phase}" for phase in PHASE_ORDER],
    )

    print("\nConfusion matrix:")
    print(confusion_df)

    print_section("RULE EXTRACTION")
    rules_df = extract_leaf_rules(main_result["model"])

    print("\nTotal leaf rules extracted:")
    print(len(rules_df))

    if not rules_df.empty:
        print("\nRules per predicted phase:")
        print(rules_df["predicted_phase"].value_counts())
        print("\nLeaf sample statistics:")
        print(rules_df["samples"].describe())

    important_rules = (
        rules_df
        .sort_values(["predicted_phase", "samples", "purity"], ascending=[True, False, False])
        .groupby("predicted_phase", group_keys=False)
        .head(10)
        .copy()
        if not rules_df.empty
        else pd.DataFrame()
    )

    error_summaries = build_error_summaries(main_result["categorized_errors"])

    save_csv(main_summary, MODEL_DIR / f"{OUTPUT_PREFIX}_main_model_summary.csv")
    save_csv(main_result["importances"], MODEL_DIR / f"{OUTPUT_PREFIX}_main_model_importances.csv")
    save_csv(confusion_df.reset_index(names="true_label"), MODEL_DIR / f"{OUTPUT_PREFIX}_confusion_matrix.csv")
    save_csv(main_result["errors"], MODEL_DIR / f"{OUTPUT_PREFIX}_raw_errors.csv")
    save_csv(build_categorized_errors_export(main_result["categorized_errors"]), MODEL_DIR / f"{OUTPUT_PREFIX}_categorized_errors.csv")
    save_csv(error_summaries["error_category_counts"], MODEL_DIR / f"{OUTPUT_PREFIX}_error_category_counts.csv")
    save_csv(error_summaries["error_type_counts"], MODEL_DIR / f"{OUTPUT_PREFIX}_error_type_counts.csv")
    save_csv(error_summaries["phase_confusion_counts"], MODEL_DIR / f"{OUTPUT_PREFIX}_phase_confusion_counts.csv")
    save_csv(error_summaries["errors_by_dtz_abs"], MODEL_DIR / f"{OUTPUT_PREFIX}_errors_by_dtz_abs.csv")
    save_csv(error_summaries["errors_by_mate_moves"], MODEL_DIR / f"{OUTPUT_PREFIX}_errors_by_mate_moves.csv")
    save_csv(error_summaries["error_flag_counts"], MODEL_DIR / f"{OUTPUT_PREFIX}_error_flag_counts.csv")
    save_csv(error_summaries["error_severity_counts"], MODEL_DIR / f"{OUTPUT_PREFIX}_error_severity_counts.csv")
    save_csv(rules_df, MODEL_DIR / f"{OUTPUT_PREFIX}_all_leaf_rules.csv")
    save_csv(important_rules, MODEL_DIR / f"{OUTPUT_PREFIX}_important_rules.csv")

    with open(MODEL_DIR / f"{OUTPUT_PREFIX}_tree_rules.txt", "w", encoding="utf-8") as file:
        file.write(main_result["rules_text"])

    write_summary_report(
        model_dir=MODEL_DIR,
        main_result=main_result,
        cv=cv,
        stability_summary=stability_summary,
        important_rules=important_rules,
        selected_main_depth=selected_main_depth,
        selected_main_class_weight=selected_main_class_weight,
        selection_info=selection_info,
    )

    print_section("FILES SAVED")
    for path in sorted(MODEL_DIR.glob(f"{OUTPUT_PREFIX}_*")):
        print(path)


if __name__ == "__main__":
    main()
