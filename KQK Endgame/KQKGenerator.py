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

KQK_OUTPUT_ROOT = Path(r"C:\Users\Κωνσταντίνος\Desktop\KQK Dataset")
DATASET_DIR = KQK_OUTPUT_ROOT / "01_dataset"
DATASET_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DATASET_CSV = DATASET_DIR / "kqk_exhaustive_dtz_dataset.csv"
OUTPUT_OVERVIEW_CSV = DATASET_DIR / "kqk_dataset_overview.csv"

CSV_SEPARATOR = ";"
DECIMAL_PLACES = 4
FLOAT_FORMAT = f"%.{DECIMAL_PLACES}f"

pd.set_option("display.float_format", lambda value: f"{value:.{DECIMAL_PLACES}f}")

SIDE_TO_MOVE = chess.WHITE
ONLY_WINNING_FOR_SIDE_TO_MOVE = True


# ============================================================
# PRINTING / SAVING UTILITIES
# ============================================================

def print_section(title: str, width: int = 90) -> None:
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def format_overview_value(value) -> str:
    """Format mixed overview values so Excel does not misread decimal dots as thousands."""

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
        encoding="utf-8-sig",
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
        encoding="utf-8-sig",
        sep=CSV_SEPARATOR,
    )


# ============================================================
# BASIC DISTANCE UTILITIES
# ============================================================

def chebyshev_distance(sq1: chess.Square, sq2: chess.Square) -> int:
    return max(
        abs(chess.square_file(sq1) - chess.square_file(sq2)),
        abs(chess.square_rank(sq1) - chess.square_rank(sq2)),
    )


def manhattan_distance(sq1: chess.Square, sq2: chess.Square) -> int:
    return (
        abs(chess.square_file(sq1) - chess.square_file(sq2))
        + abs(chess.square_rank(sq1) - chess.square_rank(sq2))
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


def wk_wq_distance(wk: chess.Square, wq: chess.Square) -> int:
    return chebyshev_distance(wk, wq)



# ============================================================
# BOARD CREATION / LEGALITY
# ============================================================

def make_kqk_board(
    wk: chess.Square,
    wq: chess.Square,
    bk: chess.Square,
    side_to_move: chess.Color,
) -> Optional[chess.Board]:
    """Create a legal KQK board: White King + White Queen vs Black King."""

    if len({wk, wq, bk}) < 3:
        return None

    if kings_adjacent(wk, bk):
        return None

    board = chess.Board(None)
    board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(wq, chess.Piece(chess.QUEEN, chess.WHITE))
    board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
    board.turn = side_to_move

    if not board.is_valid():
        return None

    # For white-to-move datasets, reject positions where black is already in check.
    # Such positions would imply that white had just moved and left the turn incorrectly.
    other_side = not side_to_move
    board.turn = other_side
    if board.is_check():
        return None

    board.turn = side_to_move
    return board


def get_piece_squares(board: chess.Board) -> Tuple[chess.Square, chess.Square, chess.Square]:
    wk = board.king(chess.WHITE)
    bk = board.king(chess.BLACK)
    wq_squares = list(board.pieces(chess.QUEEN, chess.WHITE))

    if wk is None or bk is None or len(wq_squares) != 1:
        raise ValueError("Invalid KQK board.")

    return wk, wq_squares[0], bk


# ============================================================
# KQK-SPECIFIC FEATURES
# ============================================================


def queen_box_dimensions(wq: chess.Square, bk: chess.Square) -> Tuple[int, int]:
    """
    Approximate queen box around the black king.

    The queen divides the board by file/rank. The returned dimensions estimate
    the remaining rectangular area on the side where the black king is located.
    """

    q_file = chess.square_file(wq)
    q_rank = chess.square_rank(wq)
    bk_file = chess.square_file(bk)
    bk_rank = chess.square_rank(bk)

    if bk_file < q_file:
        box_width = q_file
    elif bk_file > q_file:
        box_width = 7 - q_file
    else:
        box_width = 8

    if bk_rank < q_rank:
        box_height = q_rank
    elif bk_rank > q_rank:
        box_height = 7 - q_rank
    else:
        box_height = 8

    return box_width, box_height


def squares_around(square: chess.Square) -> List[chess.Square]:
    file_idx = chess.square_file(square)
    rank_idx = chess.square_rank(square)
    result: List[chess.Square] = []

    for df in (-1, 0, 1):
        for dr in (-1, 0, 1):
            if df == 0 and dr == 0:
                continue
            nf = file_idx + df
            nr = rank_idx + dr
            if 0 <= nf <= 7 and 0 <= nr <= 7:
                result.append(chess.square(nf, nr))

    return result


def queen_controls_bk_zone_count(board: chess.Board) -> int:
    """Count how many neighboring squares around black king are attacked by the queen."""
    wk, wq, bk = get_piece_squares(board)
    count = 0

    for sq in squares_around(bk):
        if sq in {wk, wq}:
            continue
        if wq in board.attackers(chess.WHITE, sq):
            count += 1

    return count


def black_legal_moves_if_black_to_move(board: chess.Board) -> int:
    """Helper: count of black king moves if it were black's turn (no mutation)."""
    temp = board.copy(stack=False)
    temp.turn = chess.BLACK
    return temp.legal_moves.count()


def stalemate_risk_level(board: chess.Board) -> int:
    """
    Stalemate risk for the white side to convert.

    Returns:
    - 2 (critical): it is currently white to move, but if black had the move
      he would have zero legal moves and not be in check. This is the classic
      KQK stalemate trap that white must avoid.
    - 1 (moderate): black king has at most 2 legal moves AND is on the edge or
      near a corner — a position where careless queen moves often create stalemate.
    - 0 (none): otherwise.
    """
    wk, wq, bk = get_piece_squares(board)

    # Critical risk: if it were black to move, would it be stalemate (not mate)?
    if board.turn == chess.WHITE:
        temp = board.copy(stack=False)
        temp.turn = chess.BLACK
        if temp.legal_moves.count() == 0 and not temp.is_check():
            return 2

    # Moderate risk: low mobility + black king cornered/edge
    bk_moves = black_legal_moves_if_black_to_move(board)
    bk_edge = distance_to_nearest_edge(bk)
    bk_corner = distance_to_nearest_corner(bk)

    if bk_moves <= 2 and (bk_edge == 0 or bk_corner <= 1):
        return 1

    return 0



def extract_kqk_features(board: chess.Board) -> Dict:
    wk, wq, bk = get_piece_squares(board)

    wk_file = chess.square_file(wk)
    wk_rank = chess.square_rank(wk)
    wq_file = chess.square_file(wq)
    wq_rank = chess.square_rank(wq)
    bk_file = chess.square_file(bk)
    bk_rank = chess.square_rank(bk)

    box_width, box_height = queen_box_dimensions(wq, bk)

    return {
        "fen": board.fen(),

        # Black king restriction / board geography
        "bk_legal_moves_if_black_to_move": black_legal_moves_if_black_to_move(board),
        "bk_distance_to_corner": distance_to_nearest_corner(bk),
        "bk_distance_to_edge": distance_to_nearest_edge(bk),

        # White king location / cooperation
        "wk_distance_to_edge": distance_to_nearest_edge(wk),
        "wk_wq_distance": wk_wq_distance(wk, wq),
        "wk_bk_chebyshev_distance": chebyshev_distance(wk, bk),
        "wk_bk_manhattan_distance": manhattan_distance(wk, bk),

        # Queen relation to black king
        "wq_bk_chebyshev_distance": chebyshev_distance(wq, bk),
        "queen_box_area": box_width * box_height,
        "queen_controls_bk_zone_count": queen_controls_bk_zone_count(board),
        "stalemate_risk_level": stalemate_risk_level(board),


        # Raw coordinates for diagnostics
        "wk_file": wk_file,
        "wk_rank": wk_rank,
        "wq_file": wq_file,
        "wq_rank": wq_rank,
        "bk_file": bk_file,
        "bk_rank": bk_rank,
    }


# ============================================================
# DTZ LABELING
# ============================================================

def assign_dtz_bucket(dtz_abs: int) -> str:
    """
    Convert absolute DTZ into KQK conversion buckets.

    KQK is usually much shorter than KRK, so the buckets are based on approximate
    mate-move distance.
    """
    mate_moves = (dtz_abs + 1) // 2

    if mate_moves <= 1:
        return "immediate_mate"
    if mate_moves <= 3:
        return "short_conversion"
    if mate_moves <= 6:
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
    rows = [
        {"metric": "candidate_positions_checked", "value": candidate_positions_checked},
        {"metric": "legal_positions_kept_before_wdl_filter", "value": legal_positions_kept},
        {"metric": "tablebase_probe_failures", "value": tablebase_probe_failures},
        {"metric": "final_dataset_rows", "value": len(df)},
        {"metric": "side_to_move", "value": "white" if SIDE_TO_MOVE == chess.WHITE else "black"},
        {"metric": "only_winning_for_side_to_move", "value": ONLY_WINNING_FOR_SIDE_TO_MOVE},
    ]

    if not df.empty:
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

        for value, count in df["stalemate_risk_level"].value_counts().sort_index().items():
            rows.append({"metric": f"stalemate_risk_level_count_{value}", "value": int(count)})


    return pd.DataFrame(rows)


# ============================================================
# DATASET BUILDING
# ============================================================

def build_kqk_dtz_dataset(
    tablebase_path: str = TABLEBASE_PATH,
    side_to_move: chess.Color = SIDE_TO_MOVE,
    only_winning_for_side_to_move: bool = ONLY_WINNING_FOR_SIDE_TO_MOVE,
) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    candidate_positions_checked = 0
    legal_positions_seen = 0
    tablebase_probe_failures = 0

    with chess.syzygy.open_tablebase(tablebase_path) as tablebase:
        for wk in chess.SQUARES:
            for wq in chess.SQUARES:
                if wq == wk:
                    continue

                for bk in chess.SQUARES:
                    candidate_positions_checked += 1

                    if candidate_positions_checked % 10000 == 0:
                        print(
                            f"Checked {candidate_positions_checked} candidate positions, "
                            f"kept {len(rows)} rows..."
                        )

                    if bk in {wk, wq}:
                        continue

                    board = make_kqk_board(
                        wk=wk,
                        wq=wq,
                        bk=bk,
                        side_to_move=side_to_move,
                    )
                    if board is None:
                        continue

                    legal_positions_seen += 1
                    tb_values = probe_tablebase_values(board, tablebase)

                    if tb_values is None:
                        tablebase_probe_failures += 1
                        continue

                    if only_winning_for_side_to_move and tb_values["wdl"] <= 0:
                        continue

                    row = extract_kqk_features(board)
                    row.update(tb_values)
                    rows.append(row)

    metadata = {
        "candidate_positions_checked": candidate_positions_checked,
        "legal_positions_kept": legal_positions_seen,
        "tablebase_probe_failures": tablebase_probe_failures,
    }

    return pd.DataFrame(rows), metadata


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print_section("KQK DATASET GENERATION")
    print("Tablebase path:")
    print(TABLEBASE_PATH)
    print("\nOutput root:")
    print(KQK_OUTPUT_ROOT)
    print("\nDataset output:")
    print(OUTPUT_DATASET_CSV)
    print("\nSide to move:")
    print("White" if SIDE_TO_MOVE == chess.WHITE else "Black")
    print("\nOnly winning for side to move:")
    print(ONLY_WINNING_FOR_SIDE_TO_MOVE)

    df, metadata = build_kqk_dtz_dataset(
        tablebase_path=TABLEBASE_PATH,
        side_to_move=SIDE_TO_MOVE,
        only_winning_for_side_to_move=ONLY_WINNING_FOR_SIDE_TO_MOVE,
    )

    print_section("DATASET CREATED")
    print("\nDataset shape:")
    print(df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    if df.empty:
        print("\nWARNING: Dataset is empty. Nothing will be saved.")
        return

    print("\nWDL distribution:")
    print(df["wdl"].value_counts().sort_index())
    print("\nDTZ bucket distribution:")
    print(df["dtz_bucket"].value_counts())
    print("\nDTZ statistics:")
    print(df["dtz_abs"].describe().round(DECIMAL_PLACES))
    print("\nMate-move distance distribution:")
    print(df["mate_moves"].value_counts().sort_index())

    feature_cols = [col for col in df.columns if col not in {"fen", "wdl", "dtz", "dtz_abs", "mate_moves", "dtz_bucket"}]
    print("\nFeature summary:")
    print(df[feature_cols].describe().T.round(DECIMAL_PLACES))

    sample_cols = [
        "fen", "wdl", "dtz", "dtz_abs", "mate_moves", "dtz_bucket",
        "bk_legal_moves_if_black_to_move", "bk_distance_to_corner", "bk_distance_to_edge",
        "wk_distance_to_edge", "wk_wq_distance", "wk_bk_chebyshev_distance", "wk_bk_manhattan_distance",
        "queen_box_area", "queen_controls_bk_zone_count", "stalemate_risk_level",
    ]
    existing_sample_cols = [col for col in sample_cols if col in df.columns]
    print("\nSample rows:")
    print(df[existing_sample_cols].sample(min(10, len(df)), random_state=42).to_string(index=False))

    save_csv(df, OUTPUT_DATASET_CSV)

    overview_df = build_dataset_overview(
        df=df,
        candidate_positions_checked=metadata["candidate_positions_checked"],
        legal_positions_kept=metadata["legal_positions_kept"],
        tablebase_probe_failures=metadata["tablebase_probe_failures"],
    )
    save_overview_csv(overview_df, OUTPUT_OVERVIEW_CSV)

    print_section("FILES SAVED")
    print(OUTPUT_DATASET_CSV)
    print(OUTPUT_OVERVIEW_CSV)


if __name__ == "__main__":
    main()