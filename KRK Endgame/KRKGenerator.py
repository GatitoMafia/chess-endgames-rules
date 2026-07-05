from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chess
import chess.syzygy
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

TABLEBASE_PATH = r"C:\Users\Κωνσταντίνος\Desktop\3-4-5_pieces_Syzygy\3-4-5"

KRK_OUTPUT_ROOT = Path(r"C:\Users\Κωνσταντίνος\Desktop\KRK Dataset")
DATASET_DIR = KRK_OUTPUT_ROOT / "01_dataset"

OUTPUT_DATASET_CSV = DATASET_DIR / "krk_exhaustive_dtz_dataset.csv"
OUTPUT_OVERVIEW_CSV = DATASET_DIR / "krk_dataset_overview.csv"

CSV_SEPARATOR = ";"
CSV_ENCODING = "utf-8-sig"
DECIMAL_PLACES = 4
FLOAT_FORMAT = f"%.{DECIMAL_PLACES}f"

pd.set_option("display.float_format", lambda value: f"{value:.{DECIMAL_PLACES}f}")

SIDE_TO_MOVE = chess.WHITE
ONLY_WINNING_FOR_SIDE_TO_MOVE = True
PROGRESS_REPORT_INTERVAL = 10_000

# Mating zone definition: black king on the edge OR within one square of a corner.
MATING_ZONE_CORNER_DISTANCE_THRESHOLD = 1


# ============================================================
# PRINTING / SAVING UTILITIES
# ============================================================

def print_section(title: str, width: int = 90) -> None:
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def format_overview_value(value) -> str:
    """Format mixed overview values so Excel does not misread decimal dots."""

    if isinstance(value, bool):
        return str(value)

    if isinstance(value, int):
        return str(value)

    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.{DECIMAL_PLACES}f}".replace(".", ",")

    return str(value)


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


def save_overview_csv(df: pd.DataFrame, path: Path) -> None:
    output = df.copy()

    if "value" in output.columns:
        output["value"] = output["value"].apply(format_overview_value)

    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(
        path,
        index=False,
        encoding=CSV_ENCODING,
        sep=CSV_SEPARATOR,
    )


# ============================================================
# BASIC DISTANCE UTILITIES
# ============================================================

def chebyshev_distance(square_a: chess.Square, square_b: chess.Square) -> int:
    return max(
        abs(chess.square_file(square_a) - chess.square_file(square_b)),
        abs(chess.square_rank(square_a) - chess.square_rank(square_b)),
    )


def manhattan_distance(square_a: chess.Square, square_b: chess.Square) -> int:
    return (
        abs(chess.square_file(square_a) - chess.square_file(square_b))
        + abs(chess.square_rank(square_a) - chess.square_rank(square_b))
    )


def kings_adjacent(wk: chess.Square, bk: chess.Square) -> bool:
    return chebyshev_distance(wk, bk) <= 1


def distance_to_nearest_corner(square: chess.Square) -> int:
    corners = [chess.A1, chess.H1, chess.A8, chess.H8]
    return min(chebyshev_distance(square, corner) for corner in corners)


def distance_to_nearest_edge(square: chess.Square) -> int:
    file_idx = chess.square_file(square)
    rank_idx = chess.square_rank(square)
    return min(file_idx, 7 - file_idx, rank_idx, 7 - rank_idx)


def is_strictly_between(value: int, endpoint_a: int, endpoint_b: int) -> bool:
    return min(endpoint_a, endpoint_b) < value < max(endpoint_a, endpoint_b)


# ============================================================
# BOARD CREATION AND LEGALITY
# ============================================================

def make_krk_board(
    wk: chess.Square,
    wr: chess.Square,
    bk: chess.Square,
    side_to_move: chess.Color,
) -> Optional[chess.Board]:
    """Construct a legal KRK board, or return None if the position is invalid."""

    if len({wk, wr, bk}) < 3:
        return None

    if kings_adjacent(wk, bk):
        return None

    board = chess.Board(None)
    board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(wr, chess.Piece(chess.ROOK, chess.WHITE))
    board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
    board.turn = side_to_move

    if not board.is_valid():
        return None

    # Reject positions in which the previous mover would have left the opponent in check.
    other_side = not side_to_move
    board.turn = other_side

    if board.is_check():
        return None

    board.turn = side_to_move
    return board


def get_piece_squares(board: chess.Board) -> Tuple[chess.Square, chess.Square, chess.Square]:
    wk = board.king(chess.WHITE)
    bk = board.king(chess.BLACK)
    wr_squares = list(board.pieces(chess.ROOK, chess.WHITE))

    if wk is None or bk is None or len(wr_squares) != 1:
        raise ValueError("Invalid KRK board: expected white king, white rook, black king.")

    return wk, wr_squares[0], bk


# ============================================================
# KRK FEATURE EXTRACTION
# ============================================================

def black_legal_moves_if_black_to_move(board: chess.Board) -> int:
    """Number of legal black king moves if it were Black's turn."""
    temporary_board = board.copy(stack=False)
    temporary_board.turn = chess.BLACK
    return temporary_board.legal_moves.count()


def rook_cuts_off_black_king(
    wk: chess.Square,
    wr: chess.Square,
    bk: chess.Square,
) -> int:
    """Whether the rook lies strictly between the two kings on file or rank."""
    wk_file = chess.square_file(wk)
    wk_rank = chess.square_rank(wk)
    wr_file = chess.square_file(wr)
    wr_rank = chess.square_rank(wr)
    bk_file = chess.square_file(bk)
    bk_rank = chess.square_rank(bk)

    file_cutoff = is_strictly_between(wr_file, wk_file, bk_file)
    rank_cutoff = is_strictly_between(wr_rank, wk_rank, bk_rank)

    return int(file_cutoff or rank_cutoff)


def black_king_in_mating_zone(bk: chess.Square) -> int:
    """Whether the black king is on an edge or within one square of a corner."""
    on_edge = distance_to_nearest_edge(bk) == 0
    near_corner = distance_to_nearest_corner(bk) <= MATING_ZONE_CORNER_DISTANCE_THRESHOLD
    return int(on_edge or near_corner)


def kings_in_direct_opposition(wk: chess.Square, bk: chess.Square) -> int:
    """Whether the kings are on the same file/rank with one empty square between them."""
    wk_file = chess.square_file(wk)
    wk_rank = chess.square_rank(wk)
    bk_file = chess.square_file(bk)
    bk_rank = chess.square_rank(bk)

    same_file_opposition = (wk_file == bk_file) and (abs(wk_rank - bk_rank) == 2)
    same_rank_opposition = (wk_rank == bk_rank) and (abs(wk_file - bk_file) == 2)

    return int(same_file_opposition or same_rank_opposition)


def extract_krk_features(board: chess.Board) -> Dict:
    wk, wr, bk = get_piece_squares(board)

    wk_file = chess.square_file(wk)
    wk_rank = chess.square_rank(wk)
    wr_file = chess.square_file(wr)
    wr_rank = chess.square_rank(wr)
    bk_file = chess.square_file(bk)
    bk_rank = chess.square_rank(bk)

    return {
        "fen": board.fen(),

        # Black king restriction / board geography.
        "bk_legal_moves_if_black_to_move": black_legal_moves_if_black_to_move(board),
        "bk_distance_to_corner": distance_to_nearest_corner(bk),
        "bk_distance_to_edge": distance_to_nearest_edge(bk),
        "bk_in_mating_zone": black_king_in_mating_zone(bk),

        # White king positioning and cooperation.
        "wk_distance_to_edge": distance_to_nearest_edge(wk),
        "wk_bk_chebyshev_distance": chebyshev_distance(wk, bk),
        "wk_bk_manhattan_distance": manhattan_distance(wk, bk),
        "kings_in_direct_opposition": kings_in_direct_opposition(wk, bk),

        # Rook relation to black king.
        "wr_bk_file_distance": abs(wr_file - bk_file),
        "wr_bk_rank_distance": abs(wr_rank - bk_rank),
        "rook_cuts_off_black_king": rook_cuts_off_black_king(wk, wr, bk),

        # Raw coordinates for diagnostics and raw-coordinate baselines.
        "wk_file": wk_file,
        "wk_rank": wk_rank,
        "wr_file": wr_file,
        "wr_rank": wr_rank,
        "bk_file": bk_file,
        "bk_rank": bk_rank,
    }


# ============================================================
# DTZ LABELING
# ============================================================

def assign_dtz_bucket(dtz_abs: int) -> str:
    """
    Convert absolute DTZ value into a KRK strategic conversion phase.

    DTZ buckets:
    - immediate_mate_or_near: DTZ 1-2
    - short_conversion:       DTZ 3-8
    - medium_conversion:      DTZ 9-16
    - long_conversion:        DTZ 17+
    """
    if dtz_abs <= 2:
        return "immediate_mate_or_near"
    if dtz_abs <= 8:
        return "short_conversion"
    if dtz_abs <= 16:
        return "medium_conversion"
    return "long_conversion"


def probe_tablebase_values(board: chess.Board, tablebase) -> Optional[Dict]:
    try:
        wdl = tablebase.probe_wdl(board)
        dtz = tablebase.probe_dtz(board)
    except Exception:
        return None

    dtz_abs = abs(dtz)
    mate_moves = (dtz_abs + 1) // 2

    return {
        "wdl": wdl,
        "dtz": dtz,
        "dtz_abs": dtz_abs,
        "mate_moves": mate_moves,
        "dtz_bucket": assign_dtz_bucket(dtz_abs),
    }


# ============================================================
# DATASET OVERVIEW
# ============================================================

def build_dataset_overview(
    df: pd.DataFrame,
    candidate_positions_checked: int,
    legal_positions_kept: int,
    tablebase_probe_failures: int,
) -> pd.DataFrame:
    rows: List[Dict] = [
        {"metric": "candidate_positions_checked", "value": candidate_positions_checked},
        {"metric": "legal_positions_kept_before_wdl_filter", "value": legal_positions_kept},
        {"metric": "tablebase_probe_failures", "value": tablebase_probe_failures},
        {"metric": "final_dataset_rows", "value": len(df)},
        {"metric": "side_to_move", "value": "white" if SIDE_TO_MOVE == chess.WHITE else "black"},
        {"metric": "only_winning_for_side_to_move", "value": ONLY_WINNING_FOR_SIDE_TO_MOVE},
    ]

    if df.empty:
        return pd.DataFrame(rows)

    rows.extend([
        {"metric": "dtz_abs_min", "value": float(df["dtz_abs"].min())},
        {"metric": "dtz_abs_mean", "value": float(df["dtz_abs"].mean())},
        {"metric": "dtz_abs_median", "value": float(df["dtz_abs"].median())},
        {"metric": "dtz_abs_max", "value": float(df["dtz_abs"].max())},
        {"metric": "mate_moves_min", "value": float(df["mate_moves"].min())},
        {"metric": "mate_moves_mean", "value": float(df["mate_moves"].mean())},
        {"metric": "mate_moves_median", "value": float(df["mate_moves"].median())},
        {"metric": "mate_moves_max", "value": float(df["mate_moves"].max())},
    ])

    for bucket, count in df["dtz_bucket"].value_counts().items():
        rows.append({"metric": f"dtz_bucket_count_{bucket}", "value": int(count)})

    for wdl_value, count in df["wdl"].value_counts().sort_index().items():
        rows.append({"metric": f"wdl_count_{wdl_value}", "value": int(count)})

    for binary_feature in (
        "rook_cuts_off_black_king",
        "bk_in_mating_zone",
        "kings_in_direct_opposition",
    ):
        if binary_feature not in df.columns:
            continue
        for value, count in df[binary_feature].value_counts().sort_index().items():
            rows.append({"metric": f"{binary_feature}_count_{value}", "value": int(count)})

    return pd.DataFrame(rows)


# ============================================================
# DATASET GENERATION
# ============================================================

def iterate_krk_piece_combinations():
    for wk in chess.SQUARES:
        for wr in chess.SQUARES:
            if wr == wk:
                continue
            for bk in chess.SQUARES:
                if bk == wk or bk == wr:
                    continue
                yield wk, wr, bk


def build_krk_dtz_dataset(
    tablebase_path: str = TABLEBASE_PATH,
    side_to_move: chess.Color = SIDE_TO_MOVE,
    only_winning_for_side_to_move: bool = ONLY_WINNING_FOR_SIDE_TO_MOVE,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    rows: List[Dict] = []
    candidate_positions_checked = 0
    legal_positions_kept = 0
    tablebase_probe_failures = 0

    with chess.syzygy.open_tablebase(tablebase_path) as tablebase:
        for wk, wr, bk in iterate_krk_piece_combinations():
            candidate_positions_checked += 1

            if candidate_positions_checked % PROGRESS_REPORT_INTERVAL == 0:
                print(
                    f"Checked {candidate_positions_checked:,} candidate positions, "
                    f"kept {len(rows):,} rows..."
                )

            board = make_krk_board(
                wk=wk,
                wr=wr,
                bk=bk,
                side_to_move=side_to_move,
            )

            if board is None:
                continue

            legal_positions_kept += 1
            tablebase_info = probe_tablebase_values(board, tablebase)

            if tablebase_info is None:
                tablebase_probe_failures += 1
                continue

            if only_winning_for_side_to_move and tablebase_info["wdl"] <= 0:
                continue

            features = extract_krk_features(board)
            features.update(tablebase_info)
            rows.append(features)

    statistics = {
        "candidate_positions_checked": candidate_positions_checked,
        "legal_positions_kept": legal_positions_kept,
        "tablebase_probe_failures": tablebase_probe_failures,
    }

    return pd.DataFrame(rows), statistics


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print_section("GENERATING EXHAUSTIVE KRK DTZ DATASET")
    print(f"Tablebase path:   {TABLEBASE_PATH}")
    print(f"Output folder:    {DATASET_DIR}")
    print(f"Output CSV:       {OUTPUT_DATASET_CSV}")
    print(f"Side to move:     {'white' if SIDE_TO_MOVE == chess.WHITE else 'black'}")
    print(f"Only winning STM: {ONLY_WINNING_FOR_SIDE_TO_MOVE}")

    df, statistics = build_krk_dtz_dataset()
    overview_df = build_dataset_overview(
        df=df,
        candidate_positions_checked=statistics["candidate_positions_checked"],
        legal_positions_kept=statistics["legal_positions_kept"],
        tablebase_probe_failures=statistics["tablebase_probe_failures"],
    )

    print_section("KRK DATASET CREATED")
    print("Generation statistics:")
    for key, value in statistics.items():
        print(f"- {key}: {value:,}")

    print(f"\nDataset shape: {df.shape}")
    print("\nColumns:")
    print(df.columns.tolist())

    if df.empty:
        print("\nWARNING: Dataset is empty. Nothing will be saved.")
        return

    print("\nDTZ bucket distribution:")
    print(df["dtz_bucket"].value_counts())

    print("\nWDL distribution:")
    print(df["wdl"].value_counts().sort_index())

    print("\nDTZ statistics:")
    print(df["dtz_abs"].describe().round(DECIMAL_PLACES))

    print("\nMate-move distance distribution:")
    print(df["mate_moves"].value_counts().sort_index())

    print("\nKey binary feature distributions:")
    for binary_feature in (
        "rook_cuts_off_black_king",
        "bk_in_mating_zone",
        "kings_in_direct_opposition",
    ):
        print(f"\n{binary_feature}:")
        print(df[binary_feature].value_counts().sort_index())

    save_csv(df, OUTPUT_DATASET_CSV)
    save_overview_csv(overview_df, OUTPUT_OVERVIEW_CSV)

    print_section("FILES SAVED")
    print(OUTPUT_DATASET_CSV)
    print(OUTPUT_OVERVIEW_CSV)


if __name__ == "__main__":
    main()
