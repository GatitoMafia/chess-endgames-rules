from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Optional

import chess
import chess.syzygy
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

SYZYGY_PATH = Path(r"C:\Users\Κωνσταντίνος\Desktop\3-4-5_pieces_Syzygy\3-4-5")

KPK_OUTPUT_ROOT = Path(r"C:\Users\Κωνσταντίνος\Desktop\KPK Dataset")
DATASET_DIR = KPK_OUTPUT_ROOT / "01_dataset"
DATASET_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = DATASET_DIR / "kpk_exhaustive_wdl_dataset.csv"
OVERVIEW_CSV = DATASET_DIR / "kpk_dataset_overview.csv"

USE_CANONICAL_SYMMETRY = True

CSV_SEPARATOR = ";"
CSV_ENCODING = "utf-8-sig"
DECIMAL_PLACES = 4
FLOAT_FORMAT = f"%.{DECIMAL_PLACES}f"

pd.set_option("display.float_format", lambda value: f"{value:.{DECIMAL_PLACES}f}")


# ============================================================
# PRINTING / EXPORT UTILITIES
# ============================================================

def print_section(title: str, width: int = 90) -> None:
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def convert_numeric_like_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert numeric-looking object columns, supporting both decimal comma and decimal dot.
    """
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
    """
    Read semicolon-separated final CSVs, with fallback for older comma-separated CSVs.
    """
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
    """
    Save CSV in a Windows/Excel-friendly form for Greek regional settings.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        path,
        index=index,
        encoding=CSV_ENCODING,
        sep=CSV_SEPARATOR,
        float_format=FLOAT_FORMAT,
        decimal=",",
    )


# ============================================================
# BASIC GEOMETRY UTILITIES
# ============================================================

def chebyshev_distance(square1: int, square2: int) -> int:
    return max(
        abs(chess.square_file(square1) - chess.square_file(square2)),
        abs(chess.square_rank(square1) - chess.square_rank(square2)),
    )


def kings_are_adjacent(square1: int, square2: int) -> bool:
    return chebyshev_distance(square1, square2) <= 1


def mirror_square_horizontally(square: int) -> int:
    file_idx = chess.square_file(square)
    rank_idx = chess.square_rank(square)

    return chess.square(7 - file_idx, rank_idx)


# ============================================================
# CANONICALIZATION
# ============================================================

def canonicalize_kpk_triplet(
    white_king: int,
    white_pawn: int,
    black_king: int,
) -> tuple[int, int, int]:
    """
    Use horizontal symmetry to map pawn files e-h to d-a.

    In the optimized generation loop only pawn files a-d are produced.
    This function remains as a safety guard in case the generation filter changes.
    """
    if chess.square_file(white_pawn) >= 4:
        white_king = mirror_square_horizontally(white_king)
        white_pawn = mirror_square_horizontally(white_pawn)
        black_king = mirror_square_horizontally(black_king)

    return white_king, white_pawn, black_king


def get_valid_pawn_squares() -> list[int]:
    """
    Return all generated white-pawn squares.

    White pawns on the 1st or 8th rank are excluded.
    If canonical symmetry is enabled, only files a-d are generated because e-h
    are horizontally symmetric.
    """
    valid_squares: list[int] = []

    for square in chess.SQUARES:
        file_idx = chess.square_file(square)
        rank_idx = chess.square_rank(square)

        if rank_idx in (0, 7):
            continue

        if USE_CANONICAL_SYMMETRY and file_idx > 3:
            continue

        valid_squares.append(square)

    return valid_squares


# ============================================================
# BOARD CREATION / VALIDATION
# ============================================================

def create_kpk_board(
    white_king: int,
    white_pawn: int,
    black_king: int,
    white_to_move: bool,
) -> Optional[chess.Board]:
    """
    Create a legal KPK board or return None.
    """
    if USE_CANONICAL_SYMMETRY:
        white_king, white_pawn, black_king = canonicalize_kpk_triplet(
            white_king,
            white_pawn,
            black_king,
        )

    if len({white_king, white_pawn, black_king}) < 3:
        return None

    if kings_are_adjacent(white_king, black_king):
        return None

    if chess.square_rank(white_pawn) in (0, 7):
        return None

    board = chess.Board(None)
    board.set_piece_at(white_king, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(white_pawn, chess.Piece(chess.PAWN, chess.WHITE))
    board.set_piece_at(black_king, chess.Piece(chess.KING, chess.BLACK))

    board.turn = chess.WHITE if white_to_move else chess.BLACK
    board.castling_rights = 0
    board.ep_square = None
    board.halfmove_clock = 0
    board.fullmove_number = 1

    if not board.is_valid():
        return None

    return board


def get_kpk_pieces(board: chess.Board) -> tuple[int, int, int]:
    white_king = board.king(chess.WHITE)
    black_king = board.king(chess.BLACK)
    white_pawns = list(board.pieces(chess.PAWN, chess.WHITE))
    black_pawns = list(board.pieces(chess.PAWN, chess.BLACK))

    if white_king is None or black_king is None:
        raise ValueError("Invalid KPK board: missing king.")

    if len(white_pawns) != 1 or black_pawns:
        raise ValueError("Invalid KPK board: expected White king, White pawn, Black king.")

    return white_king, white_pawns[0], black_king


# ============================================================
# KPK FEATURE FUNCTIONS
# ============================================================

def black_king_ahead_of_pawn(black_king: int, white_pawn: int) -> int:
    return int(chess.square_rank(black_king) > chess.square_rank(white_pawn))


def steps_to_promotion(white_pawn: int) -> int:
    return 8 - (chess.square_rank(white_pawn) + 1)


def black_king_inside_square_of_pawn(
    board: chess.Board,
    white_pawn: int,
    black_king: int,
) -> int:
    """
    Whether the black king is inside the square of the pawn.
    """
    promotion_square = chess.square(chess.square_file(white_pawn), 7)
    steps = steps_to_promotion(white_pawn)

    effective_steps = steps - 1 if board.turn == chess.WHITE else steps
    effective_steps = max(effective_steps, 0)

    black_king_to_promotion = chebyshev_distance(black_king, promotion_square)

    return int(black_king_to_promotion <= effective_steps)


def get_white_pawn_key_squares(white_pawn: int) -> set[int]:
    """
    Simplified KPK key-square model for a white pawn.

    Rook pawns are handled using the canonical a-file case.
    This is a practical feature approximation, not a full tablebase replacement.
    """
    pawn_file = chess.square_file(white_pawn)
    pawn_rank_1 = chess.square_rank(white_pawn) + 1

    if pawn_file == 0:
        return {chess.parse_square("b7"), chess.parse_square("b8")}

    key_squares: set[int] = set()

    candidate_files = [
        file_idx
        for file_idx in (pawn_file - 1, pawn_file, pawn_file + 1)
        if 0 <= file_idx <= 7
    ]

    if pawn_rank_1 in (2, 3, 4):
        target_ranks_1 = [pawn_rank_1 + 2]
    elif pawn_rank_1 in (5, 6):
        target_ranks_1 = [pawn_rank_1 + 1, pawn_rank_1 + 2]
    elif pawn_rank_1 == 7:
        target_ranks_1 = [8]
    else:
        return set()

    for file_idx in candidate_files:
        for rank_1 in target_ranks_1:
            if 1 <= rank_1 <= 8:
                key_squares.add(chess.square(file_idx, rank_1 - 1))

    return key_squares


def kings_have_direct_opposition(white_king: int, black_king: int) -> int:
    """
    Direct opposition: same rank/file with exactly two squares distance.
    """
    wk_file = chess.square_file(white_king)
    wk_rank = chess.square_rank(white_king)

    bk_file = chess.square_file(black_king)
    bk_rank = chess.square_rank(black_king)

    same_file = wk_file == bk_file and abs(wk_rank - bk_rank) == 2
    same_rank = wk_rank == bk_rank and abs(wk_file - bk_file) == 2

    return int(same_file or same_rank)


def kings_have_distant_opposition(white_king: int, black_king: int) -> int:
    """
    Distant opposition: same rank/file with 4 or 6 squares distance.
    """
    wk_file = chess.square_file(white_king)
    wk_rank = chess.square_rank(white_king)

    bk_file = chess.square_file(black_king)
    bk_rank = chess.square_rank(black_king)

    same_file = wk_file == bk_file and abs(wk_rank - bk_rank) in (4, 6)
    same_rank = wk_rank == bk_rank and abs(wk_file - bk_file) in (4, 6)

    return int(same_file or same_rank)


def white_king_strongly_supports_pawn(white_king: int, white_pawn: int) -> int:
    """
    Whether the white king is in a strong support zone for the pawn.

    Definition:
    - at least two ranks ahead of the pawn;
    - no more than one file away from the pawn.
    """
    rank_diff = chess.square_rank(white_king) - chess.square_rank(white_pawn)
    file_diff = abs(chess.square_file(white_king) - chess.square_file(white_pawn))

    return int(rank_diff >= 2 and file_diff <= 1)


def white_wins_key_square_race(
    board: chess.Board,
    white_king: int,
    black_king: int,
    white_pawn: int,
) -> int:
    """
    Approximate whether the white king wins the race to a pawn key square.
    """
    key_squares = get_white_pawn_key_squares(white_pawn)

    if not key_squares:
        return 0

    has_direct_opposition = kings_have_direct_opposition(white_king, black_king)

    for key_square in key_squares:
        white_distance = chebyshev_distance(white_king, key_square)
        black_distance = chebyshev_distance(black_king, key_square)

        if white_distance < black_distance:
            return 1

        if white_distance == black_distance and has_direct_opposition and board.turn == chess.BLACK:
            return 1

    return 0


def get_promotion_blockade_squares(white_pawn: int) -> set[int]:
    promotion_file = chess.square_file(white_pawn)
    promotion_rank = 7
    squares: set[int] = set()

    for file_idx in (promotion_file - 1, promotion_file, promotion_file + 1):
        for rank_idx in (promotion_rank - 1, promotion_rank):
            if 0 <= file_idx <= 7 and 0 <= rank_idx <= 7:
                squares.add(chess.square(file_idx, rank_idx))

    return squares


def black_king_can_block_promotion(
    board: chess.Board,
    white_pawn: int,
    white_king: int,
    black_king: int,
) -> int:
    """
    Approximate whether Black can create a legal promotion blockade in time.
    """
    steps = steps_to_promotion(white_pawn)
    black_moves_available = steps - 1 if board.turn == chess.WHITE else steps
    black_moves_available = max(black_moves_available, 0)

    candidate_squares = get_promotion_blockade_squares(white_pawn)
    legal_candidate_squares: set[int] = set()

    for square in candidate_squares:
        if square == white_king:
            continue

        if chebyshev_distance(square, white_king) <= 1:
            continue

        legal_candidate_squares.add(square)

    if not legal_candidate_squares:
        return 0

    minimum_black_distance = min(
        chebyshev_distance(black_king, square)
        for square in legal_candidate_squares
    )

    return int(minimum_black_distance <= black_moves_available)


# ============================================================
# TABLEBASE LABELING / FEATURE EXTRACTION
# ============================================================

def decode_kpk_result_for_white(board: chess.Board, raw_wdl: int) -> str:
    """
    Convert Syzygy WDL from side-to-move perspective to White's perspective.

    Output:
    - Win  = winning for White
    - Draw = not winning for White
    """
    if board.turn == chess.WHITE:
        return "Win" if raw_wdl > 0 else "Draw"

    return "Win" if raw_wdl < 0 else "Draw"


def build_kpk_features(board: chess.Board, tablebase) -> dict:
    white_king, white_pawn, black_king = get_kpk_pieces(board)

    white_king_pawn_distance = chebyshev_distance(white_king, white_pawn)
    black_king_pawn_distance = chebyshev_distance(black_king, white_pawn)
    kings_distance = chebyshev_distance(white_king, black_king)

    raw_wdl = tablebase.probe_wdl(board)
    result_class = decode_kpk_result_for_white(board, raw_wdl)

    return {
        "fen": board.fen(),

        # Basic state / geometry.
        "side_to_move": int(board.turn == chess.WHITE),
        "steps_to_promotion": steps_to_promotion(white_pawn),
        "is_rook_pawn": int(chess.square_file(white_pawn) == 0),
        "black_king_ahead_of_pawn": black_king_ahead_of_pawn(black_king, white_pawn),
        "white_king_pawn_distance": white_king_pawn_distance,
        "black_king_pawn_distance": black_king_pawn_distance,
        "pawn_distance_diff": black_king_pawn_distance - white_king_pawn_distance,
        "kings_distance": kings_distance,

        # Chess-informed KPK features.
        "black_king_inside_square_of_pawn": black_king_inside_square_of_pawn(
            board,
            white_pawn,
            black_king,
        ),
        "black_king_can_block_promotion": black_king_can_block_promotion(
            board,
            white_pawn,
            white_king,
            black_king,
        ),
        "white_wins_key_square_race": white_wins_key_square_race(
            board,
            white_king,
            black_king,
            white_pawn,
        ),

        # Opposition / support features.
        "kings_have_direct_opposition": kings_have_direct_opposition(
            white_king,
            black_king,
        ),
        "kings_have_distant_opposition": kings_have_distant_opposition(
            white_king,
            black_king,
        ),
        "white_king_strongly_supports_pawn": white_king_strongly_supports_pawn(
            white_king,
            white_pawn,
        ),

        # Syzygy WDL labels.
        "wdl_raw": raw_wdl,
        "result_class": result_class,
        "white_is_winning": int(result_class == "Win"),
    }


# ============================================================
# DATASET GENERATION
# ============================================================

def generate_exhaustive_kpk_dataset() -> tuple[pd.DataFrame, dict[str, int]]:
    rows: list[dict] = []
    seen_fens: set[str] = set()

    stats = {
        "checked_candidates": 0,
        "kept_rows": 0,
        "duplicate_fens_skipped": 0,
        "invalid_positions_skipped": 0,
        "tablebase_or_feature_errors": 0,
    }

    all_squares = list(chess.SQUARES)
    valid_pawn_squares = get_valid_pawn_squares()

    with chess.syzygy.open_tablebase(str(SYZYGY_PATH)) as tablebase:
        for white_king in all_squares:
            for white_pawn in valid_pawn_squares:
                for black_king in all_squares:
                    for white_to_move in (True, False):
                        stats["checked_candidates"] += 1

                        if stats["checked_candidates"] % 50_000 == 0:
                            print(
                                f"Checked {stats['checked_candidates']:,} candidates, "
                                f"kept {len(rows):,} rows..."
                            )

                        board = create_kpk_board(
                            white_king=white_king,
                            white_pawn=white_pawn,
                            black_king=black_king,
                            white_to_move=white_to_move,
                        )

                        if board is None:
                            stats["invalid_positions_skipped"] += 1
                            continue

                        fen = board.fen()

                        if fen in seen_fens:
                            stats["duplicate_fens_skipped"] += 1
                            continue

                        seen_fens.add(fen)

                        try:
                            rows.append(build_kpk_features(board, tablebase))
                        except Exception:
                            stats["tablebase_or_feature_errors"] += 1
                            continue

    stats["kept_rows"] = len(rows)

    return pd.DataFrame(rows), stats


def create_overview_dataframe(df: pd.DataFrame, stats: dict[str, int]) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for key, value in stats.items():
        records.append({
            "section": "generation_stats",
            "metric": key,
            "value": value,
        })

    records.append({
        "section": "dataset",
        "metric": "rows",
        "value": len(df),
    })

    records.append({
        "section": "dataset",
        "metric": "columns",
        "value": len(df.columns),
    })

    if df.empty:
        return pd.DataFrame(records)

    for label, count in df["result_class"].value_counts().sort_index().items():
        records.append({
            "section": "result_class",
            "metric": str(label),
            "value": int(count),
        })

    for label, count in df["white_is_winning"].value_counts().sort_index().items():
        records.append({
            "section": "white_is_winning",
            "metric": str(label),
            "value": int(count),
        })

    for label, count in df["side_to_move"].value_counts().sort_index().items():
        records.append({
            "section": "side_to_move",
            "metric": str(label),
            "value": int(count),
        })

    for feature in [
        "kings_have_direct_opposition",
        "kings_have_distant_opposition",
        "white_king_strongly_supports_pawn",
    ]:
        for label, count in df[feature].value_counts().sort_index().items():
            records.append({
                "section": feature,
                "metric": str(label),
                "value": int(count),
            })

    return pd.DataFrame(records)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print_section("GENERATING EXHAUSTIVE CANONICAL KPK WDL DATASET")

    valid_pawn_squares = get_valid_pawn_squares()
    expected_candidates = 64 * len(valid_pawn_squares) * 64 * 2

    print(f"Syzygy path: {SYZYGY_PATH}")
    print(f"Output folder: {DATASET_DIR}")
    print(f"Output CSV: {OUTPUT_CSV}")
    print(f"Overview CSV: {OVERVIEW_CSV}")
    print(f"Canonical symmetry: {USE_CANONICAL_SYMMETRY}")
    print(f"Pawn squares generated: {len(valid_pawn_squares)}")
    print(f"Expected candidate positions before filtering: {expected_candidates:,}")

    df, stats = generate_exhaustive_kpk_dataset()
    overview_df = create_overview_dataframe(df, stats)

    print_section("KPK WDL DATASET CREATED")

    print("Generation stats:")
    for key, value in stats.items():
        print(f"- {key}: {value:,}")

    print("\nDataset shape:")
    print(df.shape)

    if not df.empty:
        print("\nTarget distribution:")
        print(df["white_is_winning"].value_counts().sort_index())

        print("\nResult class distribution:")
        print(df["result_class"].value_counts())

        print("\nWDL raw distribution:")
        print(df["wdl_raw"].value_counts().sort_index())

        print("\nFeature distributions:")
        for col in [
            "kings_have_direct_opposition",
            "kings_have_distant_opposition",
            "white_king_strongly_supports_pawn",
        ]:
            print(f"\n{col}:")
            print(df[col].value_counts().sort_index())

        print("\nLabel counts:")
        for label, count in sorted(Counter(df["result_class"]).items()):
            print(f"{label}: {count}")

    save_csv(df, OUTPUT_CSV)
    save_csv(overview_df, OVERVIEW_CSV)

    print_section("FILES SAVED")
    print(OUTPUT_CSV)
    print(OVERVIEW_CSV)


if __name__ == "__main__":
    main()