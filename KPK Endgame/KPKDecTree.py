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

KPK_OUTPUT_ROOT = Path(r"C:\Users\Κωνσταντίνος\Desktop\KPK Dataset")

DATASET_DIR = KPK_OUTPUT_ROOT / "01_dataset"
MODEL_DIR = KPK_OUTPUT_ROOT / "02_model_wdl"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DATASET_CSV = DATASET_DIR / "kpk_exhaustive_wdl_dataset.csv"
OUTPUT_PREFIX = "kpk_wdl_decision_tree"

CSV_SEPARATOR = ";"
CSV_ENCODING = "utf-8-sig"
DECIMAL_PLACES = 4
FLOAT_FORMAT = f"%.{DECIMAL_PLACES}f"

pd.set_option("display.float_format", lambda value: f"{value:.{DECIMAL_PLACES}f}")

TARGET_COLUMN = "white_is_winning"
CLASS_NAMES = ["Draw", "Win"]

MAIN_SEED = 42
SEEDS = [42, 123, 456]
DEPTHS = [7, 9, 12]
CLASS_WEIGHT_OPTIONS = [None, "balanced"]

PENALTY_WEIGHT = 0.01
MIN_SAMPLES_SPLIT = 50
MIN_SAMPLES_LEAF = 25

FALLBACK_DEPTH = 12
FALLBACK_CLASS_WEIGHT: Optional[str] = None


# ============================================================
# FEATURE SET
# ============================================================

FEATURE_COLUMNS = [
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

DIAGNOSTIC_COLUMNS = [
    "fen",
    "result_class",
    "wdl_raw",
    *FEATURE_COLUMNS,
]


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def print_section(title: str, width: int = 90) -> None:
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def convert_numeric_like_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert numeric-looking object columns, supporting decimal comma and dot."""
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
    """Read semicolon-separated final CSVs, with fallback for older comma-separated CSVs."""
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
    if normalize_class_weight_value(class_weight_str) == "None":
        return None
    return str(class_weight_str)


def safe_int(value, default: int = -1) -> int:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return default
    return int(numeric)


# ============================================================
# ERROR CATEGORIZATION
# ============================================================

def classify_error_row(row: pd.Series) -> tuple[str, str]:
    """Diagnostic KPK error categories; not a formal proof of the chess cause."""
    true_label = int(row["true"])
    pred_label = int(row["pred"])
    flags: list[str] = []

    is_rook_pawn = safe_int(row.get("is_rook_pawn", 0), default=0)
    black_inside_square = safe_int(row.get("black_king_inside_square_of_pawn", 0), default=0)
    black_can_block = safe_int(row.get("black_king_can_block_promotion", 0), default=0)
    white_wins_key_race = safe_int(row.get("white_wins_key_square_race", 0), default=0)
    black_ahead = safe_int(row.get("black_king_ahead_of_pawn", 0), default=0)
    direct_opposition = safe_int(row.get("kings_have_direct_opposition", 0), default=0)
    distant_opposition = safe_int(row.get("kings_have_distant_opposition", 0), default=0)
    white_strong_support = safe_int(row.get("white_king_strongly_supports_pawn", 0), default=0)
    steps = safe_int(row.get("steps_to_promotion", -1), default=-1)
    pawn_distance_diff = safe_int(row.get("pawn_distance_diff", 0), default=0)
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
        if int(row["true"]) == 0 and int(row["pred"]) == 1
        else "false_negative_predicted_draw",
        axis=1,
    )

    return categorized


# ============================================================
# MODEL FUNCTIONS
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
            "feature": FEATURE_COLUMNS,
            "importance": clf.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

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
        "true_draw_pred_draw": int(cm[0, 0]),
        "true_draw_pred_win": int(cm[0, 1]),
        "true_win_pred_draw": int(cm[1, 0]),
        "true_win_pred_win": int(cm[1, 1]),
        "total_errors": len(errors),
        "errors": errors,
        "categorized_errors": categorized_errors,
        "importances": importances,
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=CLASS_NAMES,
            zero_division=0,
            digits=DECIMAL_PLACES,
        ),
        "rules_text": export_text(
            clf,
            feature_names=FEATURE_COLUMNS,
            decimals=DECIMAL_PLACES,
        ),
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

    clf = make_classifier(
        max_depth=max_depth,
        class_weight=class_weight,
        seed=seed,
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=seed,
    )

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
# RULE EXTRACTION / INTERPRETATION
# ============================================================

FRIENDLY_FEATURE_NAMES = {
    "side_to_move": "side to move",
    "steps_to_promotion": "steps to promotion",
    "is_rook_pawn": "rook pawn",
    "black_king_ahead_of_pawn": "black king ahead of pawn",
    "white_king_pawn_distance": "white king-pawn distance",
    "black_king_pawn_distance": "black king-pawn distance",
    "pawn_distance_diff": "black king distance minus white king distance to pawn",
    "kings_distance": "king distance",
    "black_king_inside_square_of_pawn": "black king inside square of pawn",
    "black_king_can_block_promotion": "black king can block promotion",
    "white_wins_key_square_race": "white king wins key-square race",
    "kings_have_direct_opposition": "kings have direct opposition",
    "kings_have_distant_opposition": "kings have distant opposition",
    "white_king_strongly_supports_pawn": "white king strongly supports pawn",
}

BINARY_FEATURE_SENTENCES = {
    "side_to_move": {
        1: "παίζει ο λευκός",
        0: "παίζει ο μαύρος",
    },
    "is_rook_pawn": {
        1: "το πιόνι είναι rook pawn",
        0: "το πιόνι δεν είναι rook pawn",
    },
    "black_king_ahead_of_pawn": {
        1: "ο μαύρος βασιλιάς βρίσκεται μπροστά από το πιόνι",
        0: "ο μαύρος βασιλιάς δεν βρίσκεται μπροστά από το πιόνι",
    },
    "black_king_inside_square_of_pawn": {
        1: "ο μαύρος βασιλιάς βρίσκεται μέσα στο τετράγωνο του πιονιού",
        0: "ο μαύρος βασιλιάς δεν βρίσκεται μέσα στο τετράγωνο του πιονιού",
    },
    "black_king_can_block_promotion": {
        1: "ο μαύρος βασιλιάς μπορεί να μπλοκάρει την προαγωγή",
        0: "ο μαύρος βασιλιάς δεν μπορεί να μπλοκάρει την προαγωγή",
    },
    "white_wins_key_square_race": {
        1: "ο λευκός βασιλιάς κερδίζει τον αγώνα προς τα κρίσιμα τετράγωνα",
        0: "ο λευκός βασιλιάς δεν κερδίζει τον αγώνα προς τα κρίσιμα τετράγωνα",
    },
    "kings_have_direct_opposition": {
        1: "οι βασιλιάδες έχουν άμεση opposition",
        0: "οι βασιλιάδες δεν έχουν άμεση opposition",
    },
    "kings_have_distant_opposition": {
        1: "οι βασιλιάδες έχουν distant opposition",
        0: "οι βασιλιάδες δεν έχουν distant opposition",
    },
    "white_king_strongly_supports_pawn": {
        1: "ο λευκός βασιλιάς στηρίζει ενεργά το πιόνι",
        0: "ο λευκός βασιλιάς δεν στηρίζει ενεργά το πιόνι",
    },
}


def simplify_rule_text(rule: str) -> str:
    readable = rule

    for original, replacement in FRIENDLY_FEATURE_NAMES.items():
        readable = readable.replace(original, replacement)

    return readable


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


def format_integer_threshold(operator: str, threshold: float) -> tuple[str, int]:
    threshold_as_int = int(threshold)

    if operator == "<=":
        return "το πολύ", threshold_as_int

    return "τουλάχιστον", threshold_as_int + 1


def humanize_binary_condition(
    feature: str,
    operator: str,
    threshold: float,
) -> str | None:
    if feature not in BINARY_FEATURE_SENTENCES:
        return None

    if abs(threshold - 0.5) > 1e-9:
        return None

    feature_value = 0 if operator == "<=" else 1

    return BINARY_FEATURE_SENTENCES[feature][feature_value]


def humanize_numeric_condition(
    feature: str,
    operator: str,
    threshold: float,
) -> str:
    phrase, value = format_integer_threshold(operator, threshold)

    if feature == "steps_to_promotion":
        return f"το πιόνι απέχει {phrase} {value} κινήσεις από την προαγωγή"

    if feature == "white_king_pawn_distance":
        return f"ο λευκός βασιλιάς απέχει {phrase} {value} τετράγωνα από το πιόνι"

    if feature == "black_king_pawn_distance":
        return f"ο μαύρος βασιλιάς απέχει {phrase} {value} τετράγωνα από το πιόνι"

    if feature == "pawn_distance_diff":
        if operator == "<=":
            return (
                "η διαφορά απόστασης μαύρου και λευκού βασιλιά από το πιόνι "
                f"είναι το πολύ {value}"
            )

        return (
            "η διαφορά απόστασης μαύρου και λευκού βασιλιά από το πιόνι "
            f"είναι τουλάχιστον {value}"
        )

    if feature == "kings_distance":
        return f"οι δύο βασιλιάδες απέχουν {phrase} {value} τετράγωνα μεταξύ τους"

    readable_feature = FRIENDLY_FEATURE_NAMES.get(feature, feature)

    return f"{readable_feature} {operator} {threshold:.{DECIMAL_PLACES}f}"


def humanize_condition(condition: str) -> str:
    parsed = parse_condition(condition)

    if parsed is None:
        return simplify_rule_text(condition)

    feature, operator, threshold = parsed

    binary_sentence = humanize_binary_condition(
        feature=feature,
        operator=operator,
        threshold=threshold,
    )

    if binary_sentence is not None:
        return binary_sentence

    return humanize_numeric_condition(
        feature=feature,
        operator=operator,
        threshold=threshold,
    )


def build_human_readable_rule(
    technical_rule: str,
    predicted_label: int,
) -> str:
    predicted_result = "Νίκη για τον λευκό" if predicted_label == 1 else "Ισοπαλία"

    if technical_rule == "ROOT":
        return f"Δεν υπάρχουν επιπλέον συνθήκες. Η πρόβλεψη είναι: {predicted_result}."

    conditions = technical_rule.split(" AND ")

    human_conditions = [
        humanize_condition(condition)
        for condition in conditions
    ]

    rule_body = " ΚΑΙ ".join(human_conditions)

    return f"Αν {rule_body}. Τότε η πρόβλεψη είναι: {predicted_result}."


def build_rule_summary(
    technical_rule: str,
    predicted_label: int,
) -> str:
    is_win = predicted_label == 1

    black_cannot_block = "black_king_can_block_promotion <= 0.50" in technical_rule
    black_can_block = "black_king_can_block_promotion > 0.50" in technical_rule

    black_outside_square = "black_king_inside_square_of_pawn <= 0.50" in technical_rule
    black_inside_square = "black_king_inside_square_of_pawn > 0.50" in technical_rule

    white_wins_key_race = "white_wins_key_square_race > 0.50" in technical_rule
    white_loses_key_race = "white_wins_key_square_race <= 0.50" in technical_rule

    strong_white_support = "white_king_strongly_supports_pawn > 0.50" in technical_rule
    weak_white_support = "white_king_strongly_supports_pawn <= 0.50" in technical_rule

    rook_pawn = "is_rook_pawn > 0.50" in technical_rule

    direct_opposition = "kings_have_direct_opposition > 0.50" in technical_rule
    distant_opposition = "kings_have_distant_opposition > 0.50" in technical_rule

    if is_win:
        if black_cannot_block and black_outside_square and strong_white_support:
            return (
                "Νίκη: ο μαύρος δεν προλαβαίνει blockade και ο λευκός βασιλιάς "
                "στηρίζει ενεργά το πιόνι."
            )

        if black_cannot_block and white_wins_key_race:
            return (
                "Νίκη: ο μαύρος δεν μπορεί να μπλοκάρει την προαγωγή και "
                "ο λευκός κερδίζει τα κρίσιμα τετράγωνα."
            )

        if white_wins_key_race and strong_white_support:
            return (
                "Νίκη: ο λευκός βασιλιάς κερδίζει τα κρίσιμα τετράγωνα και "
                "στηρίζει ενεργά το πιόνι."
            )

        if black_outside_square:
            return "Νίκη: ο μαύρος βασιλιάς δεν βρίσκεται μέσα στο τετράγωνο του πιονιού."

        if direct_opposition or distant_opposition:
            return "Νίκη: το μοτίβο opposition/tempo λειτουργεί υπέρ του λευκού."

        return (
            "Νίκη: ο συνδυασμός θέσης βασιλιάδων, προώθησης πιονιού και key-square "
            "motifs ευνοεί τον λευκό."
        )

    if black_can_block and white_loses_key_race:
        return (
            "Ισοπαλία: ο μαύρος προλαβαίνει να δημιουργήσει blockade και "
            "ο λευκός δεν κερδίζει τα κρίσιμα τετράγωνα."
        )

    if black_inside_square and black_can_block:
        return (
            "Ισοπαλία: ο μαύρος βρίσκεται μέσα στο τετράγωνο του πιονιού και "
            "μπορεί να μπλοκάρει την προαγωγή."
        )

    if rook_pawn and weak_white_support:
        return (
            "Ισοπαλία: το rook pawn και η ανεπαρκής στήριξη του λευκού βασιλιά "
            "ευνοούν την άμυνα."
        )

    if black_inside_square:
        return "Ισοπαλία: ο μαύρος βασιλιάς βρίσκεται μέσα στο τετράγωνο του πιονιού."

    if black_can_block:
        return "Ισοπαλία: ο μαύρος βασιλιάς μπορεί να μπλοκάρει την προαγωγή."

    if direct_opposition or distant_opposition:
        return (
            "Ισοπαλία: το μοτίβο opposition/tempo δεν επιτρέπει καθαρή μετατροπή "
            "για τον λευκό."
        )

    return (
        "Ισοπαλία: ο συνδυασμός άμυνας του μαύρου, αποστάσεων και key-square motifs "
        "δεν επιτρέπει καθαρή νίκη για τον λευκό."
    )


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
            predicted_label = int(classes[predicted_index])
            purity = float(values[predicted_index] / values.sum())

            technical_rule = " AND ".join(conditions) if conditions else "ROOT"
            predicted_result = CLASS_NAMES[predicted_label]

            rows.append(
                {
                    "predicted_label": predicted_label,
                    "predicted_result": predicted_result,
                    "samples": total,
                    "purity": round(purity, DECIMAL_PLACES),
                    "technical_rule": technical_rule,
                    "readable_rule": simplify_rule_text(technical_rule),
                    "human_readable_rule": build_human_readable_rule(
                        technical_rule=technical_rule,
                        predicted_label=predicted_label,
                    ),
                    "rule_summary": build_rule_summary(
                        technical_rule=technical_rule,
                        predicted_label=predicted_label,
                    ),
                    "draw_weight": round(float(values[0]), DECIMAL_PLACES),
                    "win_weight": round(float(values[1]), DECIMAL_PLACES),
                }
            )
            return

        feature = FEATURE_COLUMNS[tree.feature[node_id]]
        threshold = tree.threshold[node_id]

        recurse(left, conditions + [f"{feature} <= {threshold:.2f}"])
        recurse(right, conditions + [f"{feature} > {threshold:.2f}"])

    recurse(0, [])

    rules = pd.DataFrame(rows)

    if rules.empty:
        return rules

    return rules.sort_values(
        by=["predicted_result", "samples", "purity"],
        ascending=[True, False, False],
    )


# ============================================================
# EXPORT HELPERS
# ============================================================

def build_categorized_errors_export(categorized_errors: pd.DataFrame) -> pd.DataFrame:
    if categorized_errors.empty:
        return categorized_errors

    preferred_columns = [
        "fen",
        "result_class",
        "true",
        "pred",
        "error_type",
        "primary_error_category",
        "all_error_flags",
        "wdl_raw",
        "steps_to_promotion",
        "is_rook_pawn",
        "black_king_inside_square_of_pawn",
        "black_king_can_block_promotion",
        "white_wins_key_square_race",
        "kings_have_direct_opposition",
        "kings_have_distant_opposition",
        "white_king_strongly_supports_pawn",
    ]

    existing_columns = [
        col for col in preferred_columns
        if col in categorized_errors.columns
    ]

    return categorized_errors[existing_columns].copy()


def build_error_summaries(categorized_errors: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if categorized_errors.empty:
        return {
            "error_category_counts": pd.DataFrame(),
            "error_type_counts": pd.DataFrame(),
            "errors_by_steps": pd.DataFrame(),
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

    if "steps_to_promotion" in categorized_errors.columns:
        errors_by_steps = (
            categorized_errors
            .groupby(["steps_to_promotion", "error_type"])
            .size()
            .reset_index(name="count")
            .sort_values(["steps_to_promotion", "error_type"])
        )
    else:
        errors_by_steps = pd.DataFrame()

    return {
        "error_category_counts": error_category_counts,
        "error_type_counts": error_type_counts,
        "errors_by_steps": errors_by_steps,
    }


def run_multi_seed_stability(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    for seed in SEEDS:
        for depth in DEPTHS:
            for class_weight in CLASS_WEIGHT_OPTIONS:
                result = train_and_evaluate(
                    df,
                    max_depth=depth,
                    class_weight=class_weight,
                    seed=seed,
                )

                rows.append(
                    {
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
                        "false_positive_count": result["true_draw_pred_win"],
                        "false_negative_count": result["true_win_pred_draw"],
                    }
                )

                print(
                    f"seed={seed} | depth={depth} | class_weight={class_weight} | "
                    f"acc={result['test_accuracy']:.4f} | "
                    f"bal_acc={result['balanced_accuracy']:.4f} | "
                    f"macro_f1={result['macro_f1']:.4f} | "
                    f"errors={result['total_errors']}"
                )

    return pd.DataFrame(rows)


def summarize_stability(raw_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
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
            false_positive_count_mean=("false_positive_count", "mean"),
            false_positive_count_std=("false_positive_count", "std"),
            false_negative_count_mean=("false_negative_count", "mean"),
            false_negative_count_std=("false_negative_count", "std"),
            number_of_leaves_mean=("number_of_leaves", "mean"),
            number_of_leaves_std=("number_of_leaves", "std"),
        )
        .reset_index()
    )

    return summary


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

    scored["class_weight_normalized"] = scored["class_weight"].apply(
        normalize_class_weight_value
    )

    max_leaves = scored["number_of_leaves_mean"].max()

    if max_leaves is None or max_leaves <= 0:
        return pd.DataFrame()

    scored["complexity_penalty"] = penalty_weight * np.sqrt(
        scored["number_of_leaves_mean"] / max_leaves
    )

    scored["adjusted_score"] = (
        scored["balanced_accuracy_mean"] - scored["complexity_penalty"]
    )

    scored = scored.sort_values(
        by="adjusted_score",
        ascending=False,
    ).reset_index(drop=True)

    return scored


def select_optimal_main_model(
    stability_summary: pd.DataFrame,
    penalty_weight: float = PENALTY_WEIGHT,
) -> dict:
    scored = compute_depth_selection_scores(
        stability_summary=stability_summary,
        penalty_weight=penalty_weight,
    )

    if scored.empty:
        return {
            "optimal_depth": None,
            "optimal_class_weight_normalized": None,
            "message": "No stability rows available — depth selection skipped.",
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
        "false_positive_count_mean",
        "false_negative_count_mean",
        "number_of_leaves_mean",
        "number_of_leaves_std",
        "complexity_penalty",
        "adjusted_score",
    ]

    all_scores = scored[selection_columns].copy()

    return {
        "optimal_depth": int(best_row["max_depth"]),
        "optimal_class_weight_normalized": str(best_row["class_weight_normalized"]),
        "balanced_accuracy": float(best_row["balanced_accuracy_mean"]),
        "macro_f1": float(best_row["macro_f1_mean"]),
        "total_errors": float(best_row["total_errors_mean"]),
        "false_positive_count": float(best_row["false_positive_count_mean"]),
        "false_negative_count": float(best_row["false_negative_count_mean"]),
        "n_leaves": float(best_row["number_of_leaves_mean"]),
        "complexity_penalty": float(best_row["complexity_penalty"]),
        "adjusted_score": float(best_row["adjusted_score"]),
        "all_scores": all_scores,
    }


def save_depth_selection_outputs(
    model_dir: Path,
    selection_info: dict,
) -> None:
    if selection_info["optimal_depth"] is None:
        print("\nDepth selection skipped:")
        print(selection_info.get("message", "Unknown reason."))
        return

    depth_scores_path = model_dir / f"{OUTPUT_PREFIX}_depth_selection_scores.csv"
    selected_depth_path = model_dir / f"{OUTPUT_PREFIX}_selected_optimal_depth.csv"

    save_csv(
        selection_info["all_scores"],
        depth_scores_path,
    )

    selected_depth_summary = pd.DataFrame(
        [
            {
                "optimal_depth": selection_info["optimal_depth"],
                "optimal_class_weight": selection_info["optimal_class_weight_normalized"],
                "balanced_accuracy_mean": selection_info["balanced_accuracy"],
                "macro_f1_mean": selection_info["macro_f1"],
                "total_errors_mean": selection_info["total_errors"],
                "false_positive_count_mean": selection_info["false_positive_count"],
                "false_negative_count_mean": selection_info["false_negative_count"],
                "n_leaves_mean": selection_info["n_leaves"],
                "n_leaves_mean_rounded": int(round(selection_info["n_leaves"])),
                "complexity_penalty": selection_info["complexity_penalty"],
                "adjusted_score": selection_info["adjusted_score"],
                "penalty_weight": PENALTY_WEIGHT,
            }
        ]
    )

    save_csv(
        selected_depth_summary,
        selected_depth_path,
    )

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
        & (
            stability_summary["class_weight"].apply(normalize_class_weight_value)
            == selected_class_weight_normalized
        )
    ].copy()

    top_importances_text = main_result["importances"].head(10).to_string(index=False)

    if selected_stability.empty:
        selected_stability_text = "Δεν υπάρχουν διαθέσιμα αποτελέσματα σταθερότητας για το επιλεγμένο configuration."
    else:
        selected_stability_text = selected_stability.to_string(index=False)

    if important_rules.empty:
        representative_rules_text = "Δεν εξήχθησαν σημαντικοί/αντιπροσωπευτικοί κανόνες."
    else:
        rule_preview_columns = [
            "predicted_result",
            "samples",
            "purity",
            "human_readable_rule",
            "rule_summary",
        ]

        existing_columns = [
            col for col in rule_preview_columns
            if col in important_rules.columns
        ]

        representative_rules_text = important_rules[existing_columns].head(8).to_string(index=False)

    report_lines = [
        "# KPK WDL Decision Tree — Summary Report",
        "",
        "## 1. Main Model Configuration (επιλεγμένο αυτόματα μέσω adjusted_score)",
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
        f"| Classes | {CLASS_NAMES} |",
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
        "## 3. Target Definition",
        "",
        "Το KPK μοντέλο λύνει WDL binary classification:",
        "",
        "- `white_is_winning = 0`: Draw σύμφωνα με Syzygy WDL.",
        "- `white_is_winning = 1`: Win για τον λευκό σύμφωνα με Syzygy WDL.",
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
        "## 5. Confusion Matrix",
        "",
        "| True / Predicted | Predicted Draw | Predicted Win |",
        "|---|---:|---:|",
        f"| True Draw | {main_result['true_draw_pred_draw']:,} | {main_result['true_draw_pred_win']:,} |",
        f"| True Win | {main_result['true_win_pred_draw']:,} | {main_result['true_win_pred_win']:,} |",
        "",
        "## 6. Tree Complexity",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| Actual depth | {main_result['actual_depth']} |",
        f"| Number of leaves | {main_result['number_of_leaves']} |",
        f"| Total errors | {main_result['total_errors']} |",
        "",
        "## 7. Top 10 Feature Importances",
        "",
        top_importances_text,
        "",
        "## 8. Multi-Seed Stability for Selected Configuration",
        "",
        selected_stability_text,
        "",
        "## 9. Representative Extracted Rules",
        "",
        representative_rules_text,
        "",
        "Τα πλήρη αρχεία κανόνων αποθηκεύονται στα:",
        "",
        f"- `{OUTPUT_PREFIX}_all_leaf_rules.csv`",
        f"- `{OUTPUT_PREFIX}_important_rules.csv`",
        "",
        "## 10. Interpretation Note",
        "",
        "Οι εξαγόμενοι κανόνες προκύπτουν από decision tree που εκπαιδεύτηκε σε KPK θέσεις επισημασμένες από Syzygy WDL.",
        "",
        "Πρόκειται για machine-learned approximations σκακιστικών εννοιών και όχι για τυπικά σκακιστικά θεωρήματα.",
    ]

    report = "\n".join(report_lines)
    report_path = model_dir / f"{OUTPUT_PREFIX}_summary_report.md"
    report_path.write_text(report, encoding="utf-8")

    print("\nΑποθηκεύτηκε το summary report στο:")
    print(report_path)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("KPKDecTree started.")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print_section("KPK WDL DECISION TREE ANALYSIS")

    print(f"Dataset: {DATASET_CSV}")
    print(f"Output folder: {MODEL_DIR}")

    df = read_csv_auto(DATASET_CSV)
    validate_columns(df)

    print(f"\nRows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\nTarget distribution:")
    print(df[TARGET_COLUMN].value_counts().sort_index())

    print("\nTarget distribution (%):")
    print((df[TARGET_COLUMN].value_counts(normalize=True).sort_index() * 100).round(DECIMAL_PLACES))

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
            f"Falling back to FALLBACK_DEPTH={FALLBACK_DEPTH}, "
            f"FALLBACK_CLASS_WEIGHT={FALLBACK_CLASS_WEIGHT}."
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

    main_summary = pd.DataFrame(
        [
            {
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
                "false_positive_count": main_result["true_draw_pred_win"],
                "false_negative_count": main_result["true_win_pred_draw"],
            }
        ]
    )

    print(main_summary.to_string(index=False))

    print("\nClassification report:")
    print(main_result["classification_report"])

    confusion_df = pd.DataFrame(
        main_result["confusion_matrix"],
        index=["true_Draw", "true_Win"],
        columns=["pred_Draw", "pred_Win"],
    )

    print("\nConfusion matrix:")
    print(confusion_df)

    print_section("RULE EXTRACTION")

    rules_df = extract_leaf_rules(main_result["model"])

    print("\nTotal leaf rules extracted:")
    print(len(rules_df))

    if not rules_df.empty:
        print("\nRules per predicted result:")
        print(rules_df["predicted_result"].value_counts())

        print("\nLeaf sample statistics:")
        print(rules_df["samples"].describe())

    important_rules = (
        rules_df
        .sort_values(
            ["predicted_result", "samples", "purity"],
            ascending=[True, False, False],
        )
        .groupby("predicted_result", group_keys=False)
        .head(10)
        .copy()
    )

    error_summaries = build_error_summaries(main_result["categorized_errors"])

    save_csv(main_summary, MODEL_DIR / f"{OUTPUT_PREFIX}_main_model_summary.csv")
    save_csv(main_result["importances"], MODEL_DIR / f"{OUTPUT_PREFIX}_main_model_importances.csv")
    save_csv(confusion_df.reset_index(names="true_label"), MODEL_DIR / f"{OUTPUT_PREFIX}_confusion_matrix.csv")
    save_csv(main_result["errors"], MODEL_DIR / f"{OUTPUT_PREFIX}_raw_errors.csv")

    categorized_errors_for_export = build_categorized_errors_export(
        main_result["categorized_errors"]
    )

    save_csv(categorized_errors_for_export, MODEL_DIR / f"{OUTPUT_PREFIX}_categorized_errors.csv")
    save_csv(error_summaries["error_category_counts"], MODEL_DIR / f"{OUTPUT_PREFIX}_error_category_counts.csv")
    save_csv(error_summaries["error_type_counts"], MODEL_DIR / f"{OUTPUT_PREFIX}_error_type_counts.csv")
    save_csv(error_summaries["errors_by_steps"], MODEL_DIR / f"{OUTPUT_PREFIX}_errors_by_steps.csv")
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
