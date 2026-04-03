"""Page: Pick 6 — DraftKings Pick 6 MLB Player Props"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import datetime
import math

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from page_utils import (
    render_sidebar,
    add_betting_oracle_footer,
)
from retrosheet import (
    load_batting,
    load_pitching,
    load_players,
    season_batting_leaders,
    season_pitching_leaders,
)

render_sidebar(show_year_filter=False)

def get_dataframe_height(df, row_height=35, header_height=38, padding=2, max_height=600):
    """
    Calculate the optimal height for a Streamlit dataframe based on number of rows.
    
    Args:
        df (pd.DataFrame): The dataframe to display
        row_height (int): Height per row in pixels. Default: 35
        header_height (int): Height of header row in pixels. Default: 38
        padding (int): Extra padding in pixels. Default: 2
        max_height (int): Maximum height cap in pixels. Default: 600 (None for no limit)
    
    Returns:
        int: Calculated height in pixels
    
    Example:
        height = get_dataframe_height(my_df)
        st.dataframe(my_df, height=height)
    """
    num_rows = len(df)
    calculated_height = (num_rows * row_height) + header_height + padding
    
    if max_height is not None:
        return min(calculated_height, max_height)
    return calculated_height

# ─── Constants ────────────────────────────────────────────────────────────────

_CUR_YEAR = datetime.date.today().year
_SEASONS   = list(range(2020, _CUR_YEAR + 1))

# DraftKings Pick 6 batter prop categories
BATTER_PROPS = ["Hits", "Home Runs", "RBI", "Runs", "Total Bases", "Hits+Runs+RBI"]
# DraftKings Pick 6 pitcher prop categories
PITCHER_PROPS = ["Strikeouts"]

ALL_PROPS = BATTER_PROPS + PITCHER_PROPS

# Retrosheet batting column → DK prop name
_PROP_BAT_COL: dict[str, str] = {
    "Hits":         "b_h",
    "Home Runs":    "b_hr",
    "RBI":          "b_rbi",
    "Runs":         "b_r",
    "Total Bases":  "_tb",    # derived
    "Hits+Runs+RBI":"_hrr",   # derived
}
_PROP_PITCH_COL: dict[str, str] = {
    "Strikeouts": "p_k",
}

# Season avg column in leaders frame → prop
_LEADERS_BAT_COL: dict[str, str] = {
    "Hits":         ("H",   None),
    "Home Runs":    ("HR",  None),
    "RBI":          ("RBI", None),
    "Runs":         ("R",   None),
    "Total Bases":  (None,  None),   # not in leaders, computed below
    "Hits+Runs+RBI":(None,  None),   # composite
}

# Confidence tier thresholds (probability of exceeding line)
_TIER_THRESHOLDS = [(0.65, "🔥 ELITE"), (0.60, "💪 STRONG"), (0.55, "✅ GOOD"), (0.0, "⚠️ LEAN")]


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _tier(prob: float) -> str:
    for threshold, label in _TIER_THRESHOLDS:
        if prob >= threshold:
            return label
    return "⚠️ LEAN"


def _round_line(val: float) -> float:
    """Round to nearest 0.5 — standard DK line increment."""
    return round(val * 2) / 2


def _suggested_line(season_avg: float) -> float:
    """DraftKings typically sets lines ~85–90% of season avg, rounded to 0.5."""
    return max(0.5, _round_line(season_avg * 0.87))


def _df_height(df: pd.DataFrame, row_height: int = 35, header: int = 38, max_h: int = 600) -> int:
    return min(len(df) * row_height + header + 2, max_h)


@st.cache_data(show_spinner=False)
def _batter_game_logs(min_year: int = 2020, max_year: int = _CUR_YEAR) -> pd.DataFrame:
    """Per-game batter rows with derived columns (TB, H+R+RBI)."""
    df = load_batting(min_year, max_year)
    df["_tb"]  = df["b_h"] + df["b_d"] + 2 * df["b_t"] + 3 * df["b_hr"]
    df["_hrr"] = df["b_h"] + df["b_r"] + df["b_rbi"]
    return df


@st.cache_data(show_spinner=False)
def _pitcher_game_logs(min_year: int = 2020, max_year: int = _CUR_YEAR) -> pd.DataFrame:
    return load_pitching(min_year, max_year)


@st.cache_data(show_spinner=False)
def _player_registry(min_year: int = 2020, max_year: int = _CUR_YEAR) -> pd.DataFrame:
    pl = load_players(min_year, max_year)
    pl["full_name"] = pl["first"].str.strip() + " " + pl["last"].str.strip()
    return pl[["id", "full_name", "season"]].drop_duplicates()


@st.cache_data(show_spinner=False)
def _batting_leaders_cached(min_year: int = 2020, max_year: int = _CUR_YEAR) -> pd.DataFrame:
    return season_batting_leaders(min_year, max_year)


@st.cache_data(show_spinner=False)
def _pitching_leaders_cached(min_year: int = 2020, max_year: int = _CUR_YEAR) -> pd.DataFrame:
    return season_pitching_leaders(min_year, max_year, min_ip=20.0)


def _get_player_game_log(player_id: str, prop: str, season: int) -> pd.DataFrame:
    """Return a per-game DataFrame for the given player + prop column.

    If ``season`` has no rows for this player (e.g. current year not yet in
    the data store), automatically falls back to the most-recent prior season
    that does have data.
    """
    is_batter = prop in _PROP_BAT_COL

    def _load(yr: int) -> pd.DataFrame:
        if is_batter:
            df = _batter_game_logs(yr, yr)
            stat_col = _PROP_BAT_COL[prop]
        else:
            df = _pitcher_game_logs(yr, yr)
            stat_col = _PROP_PITCH_COL[prop]
        pf = df[df["id"] == player_id].copy()
        return pf, stat_col

    player_df, stat_col = _load(season)

    # Fallback: walk back up to 3 prior seasons when the requested year has no data
    for fallback_yr in range(season - 1, max(season - 4, 2019), -1):
        if not player_df.empty:
            break
        player_df, stat_col = _load(fallback_yr)

    if player_df.empty:
        return pd.DataFrame()

    # Ensure date is datetime
    if not pd.api.types.is_datetime64_any_dtype(player_df["date"]):
        player_df["date"] = pd.to_datetime(player_df["date"], errors="coerce")

    player_df = player_df.sort_values("date")
    player_df[stat_col] = pd.to_numeric(player_df[stat_col], errors="coerce")

    out = player_df[["date", "team", "opp", "vishome", stat_col]].rename(
        columns={"date": "Date", "opp": "Opponent", "vishome": "H/A", stat_col: "stat"}
    )
    out["H/A"] = out["H/A"].map({"h": "Home", "v": "Away"})
    return out.dropna(subset=["stat"]).reset_index(drop=True)


def _analyse_player(game_log: pd.DataFrame, dk_line: float):
    """Compute averages, hit rates, and MORE/LESS prediction via Normal distribution."""
    recent = game_log.tail(10)
    stat = recent["stat"]

    last_3_avg  = float(game_log.tail(3)["stat"].mean()) if len(game_log) >= 3 else float(stat.mean())
    last_5_avg  = float(game_log.tail(5)["stat"].mean()) if len(game_log) >= 5 else float(stat.mean())
    last_10_avg = float(stat.mean())
    season_avg  = float(game_log["stat"].mean())

    total_games = len(recent)
    games_over  = int((recent["stat"] > dk_line).sum())

    # Normal distribution probability estimate
    mu, sigma = float(stat.mean()), float(stat.std(ddof=1)) if len(stat) > 1 else 1.0
    sigma = max(sigma, 0.01)
    # Use weighted average: 70% last-10, 30% season
    mu_w = 0.7 * last_10_avg + 0.3 * season_avg

    from scipy.stats import norm  # lazy import — scipy available via scikit-learn deps
    prob_over = float(1 - norm.cdf(dk_line, loc=mu_w, scale=sigma))
    prob_over = max(0.05, min(0.95, prob_over))
    prob_under = 1.0 - prob_over

    if prob_over >= prob_under:
        recommendation, confidence = "MORE", prob_over
    else:
        recommendation, confidence = "LESS", prob_under

    return {
        "last_3_avg":  last_3_avg,
        "last_5_avg":  last_5_avg,
        "last_10_avg": last_10_avg,
        "season_avg":  season_avg,
        "total_games": total_games,
        "games_over":  games_over,
        "recommendation": recommendation,
        "confidence":     confidence,
        "tier":           _tier(confidence),
    }


# ─── OCR / Screenshot helpers ────────────────────────────────────────────────

# DraftKings display name → internal prop name
_DK_PROP_MAP: dict[str, str] = {
    # Full canonical labels
    "strikeouts thrown":        "Strikeouts",
    "strikeouts":               "Strikeouts",
    # OCR mis-reads of "Strikeouts Thrown" — tesseract often drops 'St' or garbles entirely
    "irkeouts thrown":          "Strikeouts",
    "rkeouts thrown":           "Strikeouts",
    "trikeouts thrown":         "Strikeouts",
    "irkeouts":                 "Strikeouts",
    "rkeouts":                  "Strikeouts",
    "trikeouts":                "Strikeouts",
    "serkeoute twrown":         "Strikeouts",
    "serkeoute":                "Strikeouts",
    "strkeoute throw":          "Strikeouts",
    "strkeoute":                "Strikeouts",
    "erkeoute":                 "Strikeouts",
    "trrown":                   "Strikeouts",
    # Hits+Runs+RBI variants including OCR garble
    "rune + r":                 "Hits+Runs+RBI",
    "runs rls":                 "Hits+Runs+RBI",   # 'te Runs Rls' after garble-prefix strip
    "runs rs":                  "Hits+Runs+RBI",   # 'tt Runs Rs' after garble-prefix strip
    # Hits + Runs + RBIs variants
    "hits + runs + rbis":       "Hits+Runs+RBI",
    "hits+runs+rbis":           "Hits+Runs+RBI",
    "hits + runs + rbi":        "Hits+Runs+RBI",
    "hits + runs":              "Hits+Runs+RBI",
    "runs + rbis":              "Hits+Runs+RBI",
    "runs + rbi":               "Hits+Runs+RBI",
    # Home runs
    "home runs":                "Home Runs",
    "home run":                 "Home Runs",
    # Total bases
    "total bases (from hits)":  "Total Bases",
    "total bases from hits":    "Total Bases",
    "total bases":              "Total Bases",
    "tal bases (from hits)":    "Total Bases",   # OCR drops 'To'
    "tal bases (from hs)":      "Total Bases",   # OCR drops 'To' + truncates 'hits'
    "tal bases":                "Total Bases",   # OCR drops 'To'
    # Hits / RBI / Runs
    "hits":                     "Hits",
    "rbis":                     "RBI",
    "rbi":                      "RBI",
    "runs scored":              "Runs",
    "runs":                     "Runs",
}


def _ocr_available() -> bool:
    """Return True only when both pytesseract and the Tesseract binary are present."""
    try:
        import shutil
        import pytesseract

        # Explicitly set the binary path so Streamlit (which may inherit a
        # different PATH from its launcher) can always find the executable.
        _exe = shutil.which("tesseract") or r"C:\Program Files\Tesseract-OCR\tesseract.EXE"
        pytesseract.pytesseract.tesseract_cmd = _exe
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


@st.cache_data(show_spinner=False)
def _parse_dk_screenshot(image_bytes: bytes) -> tuple[list[dict], str]:
    """
    Run Tesseract OCR on a DraftKings Pick 6 screenshot and extract player props.

    Strategy
    --------
    DK cards sit in a 3-column grid.  We split the image into thirds horizontally,
    OCR each column separately (better line ordering per card), then merge results.

    Returns
    -------
    (picks, raw_ocr_text)
    Each pick dict has keys: display_name, first_initial, last_name, line, prop
    """
    import re
    import io
    import shutil

    try:
        import pytesseract
        from PIL import Image, ImageEnhance
        # Ensure the binary path is explicit for the Streamlit process
        _exe = shutil.which("tesseract") or r"C:\Program Files\Tesseract-OCR\tesseract.EXE"
        pytesseract.pytesseract.tesseract_cmd = _exe
    except ImportError:
        return [], "pytesseract / Pillow not installed."

    img = Image.open(io.BytesIO(image_bytes))
    W, H = img.size

    # DK cards sit in a 3-column grid — OCR each column independently for correct line order.
    columns = [
        img.crop((0,        0, W // 3,      H)),
        img.crop((W // 3,   0, 2 * W // 3,  H)),
        img.crop((2 * W//3, 0, W,            H)),
    ]

    def _ocr_col(col_img: "Image.Image") -> str:
        grey = col_img.convert("L")
        grey = ImageEnhance.Contrast(grey).enhance(2.5)
        # PSM 6 = single uniform block; more reliable than PSM 4 for card grids
        return pytesseract.image_to_string(grey, config="--psm 6 --oem 1")

    raw_parts = [_ocr_col(c) for c in columns]
    raw_text  = "\n---col---\n".join(raw_parts)

    # ── Name pattern ────────────────────────────────────────────────────────
    # Tolerant: accept any leading junk (team-logo OCR artifacts) before "X. LastName"
    # Position (SP/OF/etc.) is optional — some DK layouts omit it.
    _POSITIONS = r"(?:SP|RP|OF|1B|2B|3B|SS|C|DH|P|CF|LF|RF)"
    name_pat = re.compile(
        r'([A-Z])\.\s{0,3}([A-Z][A-Za-z\u00C0-\u017E]{1,20}(?:[\s\-][A-Z][A-Za-z]{1,15})?)'
        r'(?:\s+' + _POSITIONS + r')?',
    )

    # ── Line number pattern ─────────────────────────────────────────────────
    # OCR lines often look like "0.5 +", "= 7.5 +", "+ 8 5.5 +".
    _line_search = re.compile(r'\b(\d{1,2}\.\d|\d{1,2})\b')
    _time_pat    = re.compile(r'\b\d{1,2}:\d{2}\b')  # strip "52:21", "12:21" before matching

    # Words that indicate a game-context line (NOT a prop line number)
    _SKIP_WORDS = frozenset({"pm", "am", "today", "bot", "top", "current", "curent",
                              "starts", "start", "stants", "inning", "more", "less", "locked",
                              "srkeouts", "irkeouts", "strikeouts"})

    def _extract_line(seg: str) -> float | None:
        # Skip game-matchup lines — they always contain "@" (e.g. "NYY @ SEA", "0@ HOU")
        # and prop lines never do.
        if "@" in seg:
            return None
        # Skip other game-context rows (scores, status, timestamps)
        seg_lower = seg.lower()
        if any(w in seg_lower for w in _SKIP_WORDS):
            return None
        # Strip clock-time patterns (e.g. "12:21") before number extraction so
        # countdown timers don't get confused for prop lines.
        seg = _time_pat.sub('', seg)
        nums = [float(m) for m in _line_search.findall(seg) if 0.5 <= float(m) <= 20.0]
        # Recover OCR-dropped decimals: "6.5" sometimes OCRs as "65", "4.5" as "45".
        # Match 2-digit numbers like X5 or X0 (15-95) that map to valid Pick 6 lines.
        for raw in re.findall(r'\b([1-9][05])\b', seg):
            v = int(raw) / 10
            if 0.5 <= v <= 9.5 and v not in nums:
                nums.append(v)
        if not nums:
            return None
        # Prefer fractional values (x.5) — typical DK Pick 6 lines
        frac = [v for v in nums if v != int(v)]
        return frac[0] if frac else nums[0]

    # Fallback name pattern: just "LastName POSITION" when initial dot is missing
    # re.IGNORECASE lets 'oF', 'sp', etc. match; looser boundary catches 'oF7ss'
    fallback_name_pat = re.compile(
        r'(?<![A-Za-z])([A-Z][a-z]{2,15}(?:\s[A-Z][a-z]{2,10})?)'
        r'\s+(?:SP|RP|OF|1B|2B|3B|SS|C\b|DH|CF|LF|RF)',
        re.IGNORECASE,
    )
    # Softer fallback: "Freeman 78" — name followed by 2-3 digit number (jersey/ownership %)
    fallback_num_pat = re.compile(
        r'(?<![A-Za-z\.])([A-Z][a-z]{3,15})\s+\d{2,3}\b'
    )
    # Ampersand-prefix: "& Montgomery" — name after & with no position token
    ampersand_name_pat = re.compile(
        r'&\s+([A-Z][a-z]{3,15})\b'
    )
    _at_initial  = re.compile(r'@([A-Z])\b')              # "@A" → "A."
    _hyphen_init = re.compile(r'\b([A-Z])-([A-Z][a-z])')   # "R-Greene" → "R. Greene"
    _sp_garble   = re.compile(r'\bS\?(?!\w)|\$\?')       # "S?" or "$?" → "SP" (OCR noise for SP)
    _title_la    = re.compile(r'\bLa\s+([a-z]{3,15})\b')  # "La cruz" → "La Cruz"
    # Strip lowercase OCR junk prefix immediately before a capital name/initial,
    # e.g. "ky FeFreemman" → "FeFreemman", "AY forte" → "Forte"
    _garble_prefix = re.compile(r'(?:^|\s)[a-z]{1,3}\s+([A-Z][A-Za-z]{2,15})\b')
    # Unwrap CamelCase-merged OCR tokens, e.g. "FeFreemman" → "Freemman"
    _camel_split   = re.compile(r'\b[A-Z][a-z]{0,3}([A-Z][a-z]{3,15})\b')
    _pos_strip   = re.compile(r'\s+(?:SP|RP|OF|1B|2B|3B|SS|DH|CF|LF|RF|C)$', re.IGNORECASE)

    def _preprocess(line: str) -> str:
        line = _at_initial.sub(lambda m: m.group(1) + '.', line)
        line = _hyphen_init.sub(lambda m: m.group(1) + '. ' + m.group(2), line)
        line = _sp_garble.sub('SP', line)
        line = _title_la.sub(lambda m: 'La ' + m.group(1).capitalize(), line)
        # Capitalise words that OCR lowercased after a junk prefix, e.g. "ky forte" → "Forte"
        line = _garble_prefix.sub(lambda m: ' ' + m.group(1), line)
        # Unwrap CamelCase-merged OCR tokens, e.g. "FeFreemman" → "Freemman"
        line = _camel_split.sub(lambda m: m.group(1), line)
        return line.strip()

    def _find_all_names(line: str) -> list[tuple[str, str]]:
        """All (initial, last_name) tokens on this line, sorted by character position."""
        results: list[tuple[str, str, int, int]] = []  # (initial, name, start, end)
        for nm in name_pat.finditer(line):
            fi = nm.group(1).upper()
            ln = _pos_strip.sub('', nm.group(2).strip()).strip()
            if len(ln) >= 2 and not ln.isupper():
                results.append((fi, ln, nm.start(), nm.end()))
        # Always run position fallback; skip spans already covered by name_pat
        for nm in fallback_name_pat.finditer(line):
            parts = nm.group(1).strip().split()
            ln = parts[-1]
            if len(ln) >= 2:
                s, e = nm.start(), nm.end()
                if not any(s < r[3] and e > r[2] for r in results):
                    results.append(('?', ln, s, e))
        # Soft fallback: "Freeman 78" — name followed by a 2-3 digit number
        for nm in fallback_num_pat.finditer(line):
            ln = nm.group(1).strip()
            # Reject common non-name words that happen to precede a number
            if len(ln) >= 4 and ln.lower() not in _SKIP_WORDS:
                s, e = nm.start(), nm.end()
                if not any(s < r[3] and e > r[2] for r in results):
                    results.append(('?', ln, s, e))
        # Last-resort: after CamelCase splitting the line may be a bare Name with
        # no position or number context (e.g. "Freemman" after "FeFreemman" split).
        # Only fire when nothing else matched AND the line is a single capitalised word
        # that is in the OCR alias table (i.e. it's a known garble, not random text).
        if not results:
            lone = re.fullmatch(r'([A-Z][a-z]{3,15})', line.strip())
            if lone and lone.group(1).lower() in _OCR_ALIAS:
                results.append(('?', lone.group(1), 0, len(lone.group(1))))
        # Ampersand prefix: "& Montgomery"
        for nm in ampersand_name_pat.finditer(line):
            ln = nm.group(1).strip()
            s, e = nm.start(), nm.end()
            if not any(s < r[3] and e > r[2] for r in results):
                results.append(('?', ln, s, e))
        results.sort(key=lambda r: r[2])
        return [(r[0], r[1]) for r in results]

    # Sort prop map by key length desc so "hits + runs + rbis" wins over "hits"
    sorted_props = sorted(_DK_PROP_MAP.items(), key=lambda kv: -len(kv[0]))

    picks: list[dict] = []
    seen: set[str] = set()  # deduplicate by (name, prop, line)

    for col_text in raw_parts:
        lines = [_preprocess(ln.strip()) for ln in col_text.split("\n") if ln.strip()]
        i = 0
        while i < len(lines):
            name_hits = _find_all_names(lines[i])
            if not name_hits:
                i += 1
                continue

            for idx, (first_initial, last_name) in enumerate(name_hits):
                pick: dict = {
                    "display_name":  f"{first_initial}. {last_name}" if first_initial != "?" else last_name,
                    "first_initial": first_initial,
                    "last_name":     last_name,
                    "line":          None,
                    "prop":          None,
                }
                for j in range(i + 1, min(i + 15, len(lines))):
                    seg = lines[j]
                    seg_stripped = seg.strip()

                    if pick["line"] is None:
                        extracted = _extract_line(seg_stripped)
                        if extracted is not None:
                            pick["line"] = extracted
                            continue

                    if pick["line"] is not None and pick["prop"] is None:
                        candidates = [seg_stripped.lower()]
                        if j + 1 < len(lines):
                            candidates.append((seg_stripped + " " + lines[j + 1].strip()).lower())
                        for candidate in candidates:
                            # Collect ALL matching props; Nth name on a merged line → Nth prop
                            matched: list[str] = []
                            for dk_name, internal in sorted_props:
                                if dk_name in candidate and internal not in matched:
                                    matched.append(internal)
                            if matched:
                                pick["prop"] = matched[min(idx, len(matched) - 1)]
                                break

                    if pick["line"] is not None and pick["prop"] is not None:
                        key = f"{first_initial}.{last_name}|{pick['prop']}|{pick['line']}"
                        if key not in seen:
                            seen.add(key)
                            picks.append(pick)
                        break
            i += 1

    return picks, raw_text


# Known OCR corruption aliases: garbled_last_lower → real_last_lower.
# Add entries here whenever a severely corrupted name is identified.
_OCR_ALIAS: dict[str, str] = {
    "onan":      "ohtani",
    "ohtan":     "ohtani",
    "ohtam":     "ohtani",
    "ohtant":    "ohtani",   # S.Ohtant OCR garble
    "rarper":    "harper",
    "harpe":     "harper",
    "crusz":     "cruz",
    "monty":     "montgomery",
    "kevan":     "kwan",      # S. Kwan (CLE) — common OCR garble
    "freemman":  "freeman",  # Freddie Freeman — double-m OCR artifact
    "fefreeman": "freeman",
    "wallner":   "wallner",  # M. Wallner — kept exact but normalises capitalisation
    "bens":      "benson",   # W. Benson — truncated OCR
    "forte":     "fortes",   # Jose Fortes / similar — OCR strips trailing 's'
    "haart":     "hernandez",# T. Hernandez — heavy OCR garble
}


def _match_player(pick: dict, registry: pd.DataFrame, season: int) -> tuple[str | None, str | None]:
    """
    Match a DK OCR-extracted name → (player_id, full_name).

    Strategy (in order):
      1. Apply known OCR alias table (e.g. "onan" → "ohtani")
      2. Exact word-boundary search on last name
      3. difflib fuzzy search on unique last names (threshold 0.62)
    For each candidate set, narrows by first initial when available.
    """
    import re
    from difflib import SequenceMatcher

    raw_last = pick["last_name"].lower()
    last     = _OCR_ALIAS.get(raw_last, raw_last)   # step 1: alias table
    initial  = pick["first_initial"].upper()

    # Prefer current-season rows; fall back to all seasons
    season_reg = registry[registry["season"] == season]
    if season_reg.empty:
        season_reg = registry

    def _narrow_by_initial(df: pd.DataFrame) -> pd.DataFrame:
        if initial and initial != "?":
            narrowed = df[df["full_name"].str.upper().str.startswith(initial, na=False)]
            return narrowed if not narrowed.empty else df
        return df

    def _best(df: pd.DataFrame):
        df = _narrow_by_initial(df)
        row = df.sort_values("season", ascending=False).iloc[0]
        return str(row["id"]), str(row["full_name"])

    # Step 2: exact word-boundary match
    pattern = r'\b' + re.escape(last) + r'\b'
    exact = season_reg[
        season_reg["full_name"].str.lower().str.contains(pattern, regex=True, na=False)
    ]
    if not exact.empty:
        return _best(exact)

    # Step 3: fuzzy match on distinct last names in the registry
    last_names = season_reg["full_name"].str.lower().str.split().str[-1].dropna().unique()
    scored = [
        (SequenceMatcher(None, last, ln).ratio(), ln)
        for ln in last_names
    ]
    scored.sort(reverse=True)
    if scored and scored[0][0] >= 0.72:
        best_ln = scored[0][1]
        # When no first initial is available, require the OCR name's first letter
        # matches the fuzzy-matched last name's first letter to avoid e.g.
        # "Kevan" (OCR of "Kwan") matching "Evans" instead.
        if initial == '?' and last and best_ln and last[0] != best_ln[0]:
            return None, None
        fuzzy_pat = r'\b' + re.escape(best_ln) + r'\b'
        fuzzy = season_reg[
            season_reg["full_name"].str.lower().str.contains(fuzzy_pat, regex=True, na=False)
        ]
        if not fuzzy.empty:
            return _best(fuzzy)

    return None, None


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    st.title("🎯 DraftKings Pick 6 – MLB")
    st.markdown("Analyze player props vs. DraftKings Pick 6 lines using historical game logs.")
    st.markdown("---")

    tab_calc, tab_top, tab_leaders, tab_shot = st.tabs([
        "📊 DK Pick 6 Calculator",
        "⭐ Top Picks",
        "🏆 Season Leaders",
        "📷 Screenshot Import",
    ])

    # ─── preload shared data ──────────────────────────────────────────────────
    registry = _player_registry(2020, _CUR_YEAR)

    # =========================================================================
    # TAB 1 — DK Pick 6 Calculator
    # =========================================================================
    with tab_calc:
        st.subheader("📊 DraftKings Pick 6 – Line Comparison")
        st.markdown("Search for a player, enter the DK line, and get a MORE / LESS recommendation.")

        col_search, col_prop = st.columns([2, 1])

        with col_search:
            search_term = st.text_input(
                "🔍 Search Player",
                placeholder="Type player name (e.g., Aaron Judge, Shohei Ohtani)",
                key="calc_search",
            )

        selected_id   = None
        selected_name = None
        selected_season = _CUR_YEAR

        if search_term:
            lower = search_term.lower()
            matches = registry[registry["full_name"].str.lower().str.contains(lower, na=False)].copy()
            if not matches.empty:
                # Show most recent season first, deduplicate by id
                matches = (
                    matches.sort_values("season", ascending=False)
                    .drop_duplicates("id")
                    .head(30)
                )
                player_opts = {
                    f"{row['full_name']} ({int(row['season'])})": (row["id"], int(row["season"]))
                    for _, row in matches.iterrows()
                }
                with col_search:
                    sel = st.selectbox("Select Player", list(player_opts.keys()), key="calc_player_sel")
                selected_id, selected_season = player_opts[sel]
                selected_name = sel.split("(")[0].strip()
            else:
                with col_search:
                    st.info("No players found. Try a different name.")

        with col_prop:
            prop = st.selectbox("Prop Type", ALL_PROPS, key="calc_prop")

        if selected_id and prop:
            game_log = _get_player_game_log(selected_id, prop, selected_season)

            if game_log is None or game_log.empty:
                st.warning(f"No {prop} game log data found for this player in {selected_season}.")
            else:
                season_mean = float(game_log["stat"].mean())
                sug_line    = _suggested_line(season_mean)

                dk_line = st.number_input(
                    f"DraftKings Pick 6 Line for {prop}",
                    min_value=0.5,
                    max_value=50.0,
                    value=sug_line,
                    step=0.5,
                    key="calc_line",
                    help="Enter the exact over/under line from DraftKings Pick 6",
                )

                res = _analyse_player(game_log, dk_line)

                # ── Hero card ──────────────────────────────────────────────
                rec_color = "#16a34a" if res["recommendation"] == "MORE" else "#dc2626"
                st.markdown(f"""
<div style="background: linear-gradient(135deg, #002D72 0%, #D50032 100%);
            padding: 2rem; border-radius: 15px; color: white; margin: 1rem 0;">
  <h2 style="margin: 0; font-size: 1.8rem;">{selected_name}</h2>
  <p style="margin: 0.5rem 0; font-size: 1.1rem; opacity: 0.9;">{prop} · {selected_season}</p>
  <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 1.5rem;">
    <div>
      <p style="margin: 0; font-size: 0.9rem; opacity: 0.8;">DraftKings Line</p>
      <p style="margin: 0; font-size: 2.5rem; font-weight: bold;">{dk_line:.1f}</p>
    </div>
    <div style="text-align: right;">
      <p style="margin: 0; font-size: 0.9rem; opacity: 0.8;">Recommendation</p>
      <p style="margin: 0; font-size: 2.5rem; font-weight: bold; color: #ffd700;">{res['recommendation']}</p>
      <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem;">{res['tier']}</p>
    </div>
  </div>
  <div style="margin-top: 1.5rem; padding-top: 1.5rem; border-top: 1px solid rgba(255,255,255,0.2);">
    <p style="margin: 0; font-size: 1rem;">
      Confidence: <strong>{res['confidence']:.1%}</strong> &nbsp;·&nbsp; Season avg: <strong>{res['season_avg']:.2f}</strong>
    </p>
    <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; opacity: 0.8;">
      Last {res['total_games']} games: {res['games_over']} over, {res['total_games'] - res['games_over']} under
    </p>
  </div>
</div>
                """, unsafe_allow_html=True)

                # ── Performance metrics ────────────────────────────────────
                st.markdown("### 📈 Performance Analysis")
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric(
                    "Last 3 Games Avg", f"{res['last_3_avg']:.2f}",
                    f"{res['last_3_avg'] - dk_line:+.2f} vs line",
                )
                mc2.metric(
                    "Last 5 Games Avg", f"{res['last_5_avg']:.2f}",
                    f"{res['last_5_avg'] - dk_line:+.2f} vs line",
                )
                mc3.metric(
                    "Last 10 Games Avg", f"{res['last_10_avg']:.2f}",
                    f"{res['last_10_avg'] - dk_line:+.2f} vs line",
                )
                mc4.metric(
                    "Season Avg", f"{res['season_avg']:.2f}",
                    f"{res['season_avg'] - dk_line:+.2f} vs line",
                )

                # ── Recent game log ────────────────────────────────────────
                st.markdown("### 📋 Recent Game Log (Last 10)")
                gl_disp = game_log.tail(10).copy()
                if "Date" in gl_disp.columns and pd.api.types.is_datetime64_any_dtype(gl_disp["Date"]):
                    gl_disp["Date"] = gl_disp["Date"].dt.date

                gl_disp["Result"] = gl_disp["stat"].apply(
                    lambda x: f"✅ OVER ({x:.1f})" if x > dk_line else f"❌ UNDER ({x:.1f})"
                )
                gl_disp[prop] = gl_disp["stat"].round(2)
                disp_cols = ["Date", "H/A", "Opponent", prop, "Result"]
                avail = [c for c in disp_cols if c in gl_disp.columns]
                st.dataframe(
                    gl_disp[avail].sort_values("Date", ascending=False),
                    hide_index=True,
                    width='content',
                    height=get_dataframe_height(gl_disp[avail]),
                )

                # ── Hit rate analysis ─────────────────────────────────────
                st.markdown("### 🎯 Hit Rate Analysis")
                hc1, hc2, hc3 = st.columns(3)
                with hc1:
                    l3 = game_log.tail(3)
                    o3 = int((l3["stat"] > dk_line).sum())
                    hc1.metric("Last 3 Games", f"{o3}/3 Over",
                               f"{o3/3*100:.0f}% hit rate" if len(l3) >= 3 else "—")
                with hc2:
                    l5 = game_log.tail(5)
                    o5 = int((l5["stat"] > dk_line).sum())
                    hc2.metric("Last 5 Games", f"{o5}/5 Over",
                               f"{o5/5*100:.0f}% hit rate" if len(l5) >= 5 else "—")
                with hc3:
                    tg = res["total_games"]
                    ov = res["games_over"]
                    hc3.metric(f"Last {tg} Games", f"{ov}/{tg} Over",
                               f"{ov/tg*100:.0f}% hit rate" if tg > 0 else "—")

                # ── Trend chart ───────────────────────────────────────────
                st.markdown("### 📊 Trend Chart")
                chart_df = game_log.tail(20).copy()
                chart_df["Game"] = range(1, len(chart_df) + 1)
                fig = px.bar(
                    chart_df,
                    x="Game",
                    y="stat",
                    color=chart_df["stat"].apply(lambda x: "Over" if x > dk_line else "Under"),
                    color_discrete_map={"Over": "#16a34a", "Under": "#dc2626"},
                    title=f"{selected_name} — {prop} (Last {len(chart_df)} games)",
                    labels={"stat": prop},
                )
                fig.add_hline(y=dk_line, line_dash="dash", line_color="#f59e0b",
                              annotation_text=f"DK Line {dk_line:.1f}", annotation_position="top right")
                fig.update_layout(showlegend=True, legend_title_text="vs Line")
                st.plotly_chart(fig, width='stretch')

                with st.expander("ℹ️ How This Works"):
                    st.markdown(f"""
**Probability model** — fits a Normal distribution to this player's last 10 games (weighted 70%)
and full-season average (30%) to estimate `P(stat > line)`.

**Confidence Tiers:**
| Tier | Probability | Meaning |
|------|------------|---------|
| 🔥 ELITE  | ≥ 65% | Highest-confidence pick |
| 💪 STRONG | 60–65% | Very confident |
| ✅ GOOD   | 55–60% | Solid above breakeven |
| ⚠️ LEAN   | < 55% | Lower confidence — use sparingly |

**MORE** = model expects player to *exceed* the line  
**LESS** = model expects player to *fall short*

**Suggested Line** shown by default is 87% of season avg rounded to nearest 0.5.
Always verify against the actual DraftKings line before placing a pick.
                    """)

    # =========================================================================
    # TAB 2 — Top Picks
    # =========================================================================
    with tab_top:
        st.subheader("⭐ Top Prop Picks by Category")
        st.markdown("Season average vs. suggested DK lines — sorted by confidence tier.")

        tc1, tc2, tc3 = st.columns(3)
        with tc1:
            top_prop = st.selectbox(
                "Prop Type",
                ["Hits", "Home Runs", "RBI", "Runs", "Strikeouts"],
                key="top_prop",
            )
        with tc2:
            top_season = st.selectbox("Season", sorted(_SEASONS, reverse=True), key="top_season")
        with tc3:
            top_n = st.slider("Players to show", 10, 50, 25, key="top_n")

        st.markdown("---")

        # Load appropriate leaders table
        if top_prop == "Strikeouts":
            leaders = _pitching_leaders_cached(top_season, top_season)
            if leaders.empty:
                st.warning(f"No pitching data available for {top_season}.")
            else:
                leaders = leaders[leaders["SO"] > 0].copy()
                leaders["season_avg"] = (leaders["SO"] / leaders["GS"].where(leaders["GS"] > 0, 1)).round(2)
                leaders["Sug. Line"] = leaders["season_avg"].apply(_suggested_line)
                leaders["_ratio"]   = leaders["season_avg"] / leaders["Sug. Line"].where(leaders["Sug. Line"] > 0, 1)
                leaders["Tier"] = leaders["_ratio"].apply(
                    lambda r: "🔥 ELITE" if r >= 1.20 else "💪 STRONG" if r >= 1.12 else "✅ GOOD" if r >= 1.05 else "⚠️ LEAN"
                )
                leaders = leaders.sort_values("season_avg", ascending=False).head(top_n).copy()
                leaders.insert(0, "#", range(1, len(leaders) + 1))

                t1, t2, t3 = st.columns(3)
                t1.metric("🔥 Elite",  len(leaders[leaders["Tier"] == "🔥 ELITE"]))
                t2.metric("💪 Strong", len(leaders[leaders["Tier"] == "💪 STRONG"]))
                t3.metric("✅ Good",   len(leaders[leaders["Tier"] == "✅ GOOD"]))
                st.markdown("---")

                show = ["#", "full_name", "team", "GS", "SO", "season_avg", "Sug. Line", "Tier"]
                avail = [c for c in show if c in leaders.columns]
                disp = leaders[avail].rename(columns={"full_name": "Player", "team": "Team",
                                                       "season_avg": "K/Start"})
                for c in disp.select_dtypes("float").columns:
                    disp[c] = disp[c].round(1)
                st.dataframe(disp, width='stretch', hide_index=True, height=_df_height(disp))
                st.caption(
                    f"Top {len(leaders)} K/Start candidates · {top_season} season · "
                    "Suggested Line ≈ 87% of K/start avg · Verify against live DK lines."
                )

        else:
            # Batting props
            bat_col_map = {"Hits": "H", "Home Runs": "HR", "RBI": "RBI", "Runs": "R"}
            stat_col = bat_col_map.get(top_prop)
            leaders = _batting_leaders_cached(top_season, top_season)
            if leaders.empty or stat_col is None or stat_col not in leaders.columns:
                st.warning(f"No batting data available for {top_season}.")
            else:
                leaders = leaders[leaders["PA"] >= 50].copy()
                leaders["season_avg"] = (leaders[stat_col] / leaders["PA"].where(leaders["PA"] > 0, np.nan) * 4.5).round(2)
                # Better per-game approx: stat_col / number of games (PA/4.5 ≈ games)
                g_approx = (leaders["PA"] / 4.5).clip(lower=1)
                leaders["per_game"] = (leaders[stat_col] / g_approx).round(3)
                leaders["Sug. Line"] = leaders["per_game"].apply(_suggested_line)
                leaders = leaders[leaders["Sug. Line"] >= 0.5]
                leaders["_ratio"] = leaders["per_game"] / leaders["Sug. Line"].where(leaders["Sug. Line"] > 0, 1)
                leaders["Tier"] = leaders["_ratio"].apply(
                    lambda r: "🔥 ELITE" if r >= 1.20 else "💪 STRONG" if r >= 1.12 else "✅ GOOD" if r >= 1.05 else "⚠️ LEAN"
                )
                leaders = leaders.sort_values("per_game", ascending=False).head(top_n).copy()
                leaders.insert(0, "#", range(1, len(leaders) + 1))

                t1, t2, t3 = st.columns(3)
                t1.metric("🔥 Elite",  len(leaders[leaders["Tier"] == "🔥 ELITE"]))
                t2.metric("💪 Strong", len(leaders[leaders["Tier"] == "💪 STRONG"]))
                t3.metric("✅ Good",   len(leaders[leaders["Tier"] == "✅ GOOD"]))
                st.markdown("---")

                show = ["#", "full_name", "team", "PA", stat_col, "per_game", "Sug. Line", "Tier"]
                avail = [c for c in show if c in leaders.columns]
                disp = leaders[avail].rename(columns={
                    "full_name": "Player", "team": "Team",
                    stat_col: f"{top_prop} (Season)", "per_game": f"{top_prop}/Game",
                })
                for c in disp.select_dtypes("float").columns:
                    disp[c] = disp[c].round(2)
                st.dataframe(disp, width='stretch', hide_index=True, height=_df_height(disp))
                st.caption(
                    f"Top {len(leaders)} {top_prop}/game candidates · {top_season} season · "
                    "Suggested Line ≈ 87% of per-game avg · Always verify against live DK lines."
                )

        with st.expander("ℹ️ Tier calculation"):
            st.markdown("""
| Tier | Ratio (avg ÷ line) | Meaning |
|------|-------------------|---------|
| 🔥 ELITE  | ≥ 1.20× | Player consistently exceeds this line |
| 💪 STRONG | 1.12–1.20× | Very likely to hit |
| ✅ GOOD   | 1.05–1.12× | Solid edge above breakeven |
| ⚠️ LEAN   | < 1.05× | No strong season-average edge |

**Suggested Line** is a conservative estimate (~87% of per-game average).
The actual DraftKings line may differ — always check before betting.
            """)

    # =========================================================================
    # TAB 3 — Season Leaders
    # =========================================================================
    with tab_leaders:
        st.subheader("🏆 Season Leaders")

        lc1, lc2 = st.columns(2)
        with lc1:
            l_season = st.selectbox("Season", sorted(_SEASONS, reverse=True), key="leaders_season")
        with lc2:
            l_top = st.slider("Top N", 10, 50, 20, key="leaders_top")

        bat_lead = _batting_leaders_cached(l_season, l_season)
        pit_lead = _pitching_leaders_cached(l_season, l_season)

        ltab1, ltab2, ltab3, ltab4, ltab5 = st.tabs([
            "🪄 Hits", "💣 Home Runs", "🏃 Runs", "🎯 RBI", "⚡ K (Pitcher)"
        ])

        def _show_bat_leaders(df: pd.DataFrame, stat: str, label: str, top: int):
            if df.empty or stat not in df.columns:
                st.info("No data available.")
                return
            g_approx = (df["PA"] / 4.5).clip(lower=1)
            d = df.copy()
            d["per_game"] = (d[stat] / g_approx).round(3)
            d = d.sort_values(stat, ascending=False).head(top).copy()
            d.insert(0, "#", range(1, len(d) + 1))
            # Extra context cols — exclude stat itself to avoid duplicates
            extra = [c for c in ["BA", "HR"] if c != stat and c in d.columns]
            avail_cols = [c for c in ["#", "full_name", "team", "PA", stat, "per_game"] + extra if c in d.columns]
            rename = {"full_name": "Player", "team": "Team", stat: label, "per_game": f"{label}/G", "BA": "AVG"}
            disp = d[avail_cols].rename(columns=rename)
            for c in disp.select_dtypes("float").columns:
                disp[c] = disp[c].round(3 if "AVG" in disp.columns else 1)
            st.dataframe(disp, width='stretch', hide_index=True, height=_df_height(disp))
            st.caption(f"Top {len(d)} by {label} — {l_season}")

        with ltab1:
            _show_bat_leaders(bat_lead, "H", "Hits", l_top)
        with ltab2:
            _show_bat_leaders(bat_lead, "HR", "Home Runs", l_top)
        with ltab3:
            _show_bat_leaders(bat_lead, "R", "Runs", l_top)
        with ltab4:
            _show_bat_leaders(bat_lead, "RBI", "RBI", l_top)
        with ltab5:
            if pit_lead.empty or "SO" not in pit_lead.columns:
                st.info("No pitching data available.")
            else:
                d = pit_lead.sort_values("SO", ascending=False).head(l_top).copy()
                d.insert(0, "#", range(1, len(d) + 1))
                d["K/GS"] = (d["SO"] / d["GS"].where(d["GS"] > 0, 1)).round(2)
                cols = [c for c in ["#", "full_name", "team", "GS", "IP", "SO", "K/GS", "ERA", "WHIP"] if c in d.columns]
                disp = d[cols].rename(columns={"full_name": "Player", "team": "Team"})
                for c in disp.select_dtypes("float").columns:
                    disp[c] = disp[c].round(2)
                st.dataframe(disp, width='stretch', hide_index=True, height=_df_height(disp))
                st.caption(f"Top {len(d)} by Strikeouts — {l_season}")

    # =========================================================================
    # TAB 4 — Screenshot Import
    # =========================================================================
    with tab_shot:
        st.subheader("📷 Screenshot Import — DraftKings Pick 6")
        st.markdown(
            "Upload a DraftKings Pick 6 screenshot and get instant MORE / LESS recommendations "
            "based on historical game-log statistics."
        )

        ocr_ok   = _ocr_available()
        season_s = st.selectbox(
            "Season",
            sorted(_SEASONS, reverse=True),
            key="shot_season",
            # Default to most recent complete season (prior year) since current
            # season data is unavailable until mid-April at the earliest.
            index=1 if _CUR_YEAR in _SEASONS and len(_SEASONS) > 1 else 0,
        )

        if not ocr_ok:
            st.warning(
                "⚠️ **Tesseract OCR is not installed** — automatic parsing is unavailable.  \n"
                "Install it:  \n"
                "- **Windows:** [Tesseract installer](https://github.com/UB-Mannheim/tesseract/wiki) "
                "then add to PATH  \n"
                "- **macOS:** `brew install tesseract`  \n"
                "- **Linux / Streamlit Cloud:** `apt-get install tesseract-ocr` (handled via `packages.txt`)"
            )
            st.markdown("---")
            st.markdown("### ✍️ Manual Entry")
            st.markdown("Enter each prop you see on the DraftKings card:")

            with st.form("manual_entry_form"):
                n_rows = st.number_input("Number of picks", 1, 12, 6)
                manual_picks: list[dict] = []
                for idx in range(int(n_rows)):
                    c1, c2, c3 = st.columns([3, 2, 1])
                    pname = c1.text_input(f"Player #{idx+1} name",    key=f"mp_name_{idx}")
                    pprop = c2.selectbox(f"Prop #{idx+1}",             ALL_PROPS, key=f"mp_prop_{idx}")
                    pline = c3.number_input(f"Line #{idx+1}", 0.5, 50.0, 0.5, 0.5, key=f"mp_line_{idx}")
                    if pname:
                        manual_picks.append({"display_name": pname, "line": pline, "prop": pprop})

                submitted = st.form_submit_button("🔍 Analyse Picks")

            if submitted and manual_picks:
                _render_pick_results(manual_picks, registry, season_s, from_ocr=False)
        else:
            uploaded = st.file_uploader(
                "Upload DraftKings Pick 6 screenshot",
                type=["png", "jpg", "jpeg"],
                key="shot_upload",
                help="Drag-and-drop or click to browse. PNG / JPG accepted.",
            )

            if uploaded:
                img_bytes = uploaded.read()
                st.image(img_bytes, caption="Uploaded screenshot", width='stretch')

                with st.spinner("Running OCR — parsing player cards…"):
                    parsed_picks, raw_ocr = _parse_dk_screenshot(img_bytes)

                # Always show raw OCR so we can iterate on the regex without guessing
                with st.expander("🔍 Raw OCR output (debug)", expanded=not parsed_picks):
                    st.code(raw_ocr, language="")
                    if parsed_picks:
                        st.write("**Parsed picks:**", parsed_picks)
                    # Show a sample of registry names to confirm player data is loaded
                    st.write(f"**Registry rows:** {len(registry)}  |  **Sample names:** {registry['full_name'].drop_duplicates().head(10).tolist()}")

                if not parsed_picks:
                    st.error(
                        "Could not extract any player props from this screenshot.  \n"
                        "Tips: use a full-resolution screenshot with all card text visible."
                    )
                else:
                    st.success(f"Detected **{len(parsed_picks)} props** — analysing…")
                    _render_pick_results(parsed_picks, registry, season_s, from_ocr=True)

    add_betting_oracle_footer()


def _render_pick_results(
    picks: list[dict],
    registry: pd.DataFrame,
    season: int,
    from_ocr: bool,
) -> None:
    """
    Shared renderer: match players, run analysis, display ranked results.
    `picks` items must have keys: display_name, prop, line.
    When from_ocr=True they also have first_initial / last_name.
    """
    # Progress UI
    total = max(len(picks), 1)
    progress_bar = st.progress(0)
    progress_text = st.empty()

    rows = []
    unmatched = []

    for idx, pick in enumerate(picks, start=1):
        progress_text.markdown(f"## Processing pick {idx}/{total}...")
        progress_bar.progress(int((idx - 1) / total * 100))

        if from_ocr:
            pid, fname = _match_player(pick, registry, season)
        else:
            # Manual entry: search by full text
            lower = pick["display_name"].lower()
            candidates = registry[registry["full_name"].str.lower().str.contains(lower, na=False)]
            if candidates.empty:
                pid, fname = None, None
            else:
                best = candidates.sort_values("season", ascending=False).iloc[0]
                pid, fname = str(best["id"]), str(best["full_name"])

        if not pid:
            raw_last = pick.get("last_name", "").lower()
            alias = _OCR_ALIAS.get(raw_last)
            hint = f" (tried alias → {alias})" if alias else " (no fuzzy match found)"
            unmatched.append(pick["display_name"] + hint)
            continue

        gl = _get_player_game_log(pid, pick["prop"], season)
        if gl is None or gl.empty:
            unmatched.append(f"{pick['display_name']} (no {pick['prop']} data in {season})")
            continue

        res = _analyse_player(gl, pick["line"])
        rows.append({
            "Player":      fname,
            "OCR Name":    pick["display_name"],
            "Prop":        pick["prop"],
            "DK Line":     pick["line"],
            "Rec":         res["recommendation"],
            "Confidence":  res["confidence"],
            "Tier":        res["tier"],
            "Season Avg":  round(res["season_avg"],  2),
            "Last 10 Avg": round(res["last_10_avg"], 2),
            "player_id":   pid,
        })

    progress_bar.progress(100)
    progress_text.markdown("## ✅ Processing complete")

    if not rows:
        st.warning("No players could be matched against the historical database.")
        return

    rows_sorted = sorted(rows, key=lambda x: x["Confidence"], reverse=True)

    st.markdown("### 🏆 Ranked Recommendations")

    for rank, r in enumerate(rows_sorted, 1):
        rec_color  = "#16a34a" if r["Rec"] == "MORE" else "#dc2626"
        tier_emoji = r["Tier"].split()[0]  # e.g. "🔥"
        with st.container():
            st.markdown(
                f"""
<div style="background:#f8fafc; border-left:4px solid {rec_color};
            padding:0.75rem 1rem; border-radius:8px; margin-bottom:0.5rem; color:#111827; border:1px solid #cbd5e1;">
  <span style="font-size:1.05rem; font-weight:700;">#{rank} {r['Player']}</span>
  {f'<span style="opacity:.5; font-size:.8rem;"> ← {r["OCR Name"]}</span>' if r['OCR Name'].lower() != r['Player'].lower().split()[-1].lstrip() and r['OCR Name'] != r['Player'] else ''}
  &nbsp;·&nbsp; <span style="opacity:.75;">{r['Prop']}</span>
  &nbsp;·&nbsp; Line: <strong>{r['DK Line']}</strong>
  &nbsp;&nbsp;
  <span style="color:{rec_color}; font-weight:900; font-size:1.1rem;">{r['Rec']}</span>
  &nbsp; {r['Tier']}
  &nbsp;&nbsp;
  <span style="opacity:.6; font-size:.9rem;">
    Conf: {r['Confidence']:.1%} &nbsp;|&nbsp;
    Season avg: {r['Season Avg']} &nbsp;|&nbsp;
    L10 avg: {r['Last 10 Avg']}
  </span>
</div>""",
                unsafe_allow_html=True,
            )

    # Summary table
    st.markdown("### 📋 Summary Table")
    disp_cols = ["Player", "OCR Name", "Prop", "DK Line", "Rec", "Tier", "Confidence", "Season Avg", "Last 10 Avg"]
    df_res = pd.DataFrame(rows_sorted)[disp_cols].copy()
    df_res["Confidence"] = df_res["Confidence"].apply(lambda x: f"{x:.1%}")
    st.dataframe(df_res, hide_index=True, width='stretch', height=get_dataframe_height(df_res))

    if unmatched:
        with st.expander(f"⚠️ {len(unmatched)} player(s) not matched / no data"):
            for name in unmatched:
                st.write(f"- {name}")


main()
