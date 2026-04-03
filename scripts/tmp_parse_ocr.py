"""Simulate _parse_dk_screenshot parsing against the real OCR text dump (v4 — April 2 slate)."""
import re

_AT_INITIAL    = re.compile(r'@([A-Z])\b')
_HYPHEN_INIT   = re.compile(r'\b([A-Z])-([A-Z][a-z])')
_POS_STRIP     = re.compile(r'\s+(?:SP|RP|OF|1B|2B|3B|SS|DH|CF|LF|RF|C)$', re.IGNORECASE)
_SP_GARBLE     = re.compile(r'\bS\?(?!\w)|\$\?')
_TITLE_LA      = re.compile(r'\bLa\s+([a-z]{3,15})\b')
_GARBLE_PREFIX = re.compile(r'(?:^|\s)[a-z]{1,3}\s+([A-Z][A-Za-z]{2,15})\b')
_CAMEL_SPLIT   = re.compile(r'\b[A-Z][a-z]{0,3}([A-Z][a-z]{3,15})\b')

_OCR_ALIAS = {
    "onan":"ohtani","ohtan":"ohtani","ohtam":"ohtani","ohtant":"ohtani",
    "rarper":"harper","harpe":"harper","crusz":"cruz","monty":"montgomery",
    "kevan":"kwan","freemman":"freeman","fefreeman":"freeman",
    "bens":"benson","forte":"fortes","haart":"hernandez",
}

def _preprocess(line: str) -> str:
    line = _AT_INITIAL.sub(lambda m: m.group(1) + '.', line)
    line = _HYPHEN_INIT.sub(lambda m: m.group(1) + '. ' + m.group(2), line)
    line = _SP_GARBLE.sub('SP', line)
    line = _TITLE_LA.sub(lambda m: 'La ' + m.group(1).capitalize(), line)
    line = _GARBLE_PREFIX.sub(lambda m: ' ' + m.group(1), line)
    line = _CAMEL_SPLIT.sub(lambda m: m.group(1), line)
    return line.strip()

_DK_PROP_MAP = {
    "strikeouts thrown": "Strikeouts", "strikeouts": "Strikeouts",
    "irkeouts thrown": "Strikeouts", "rkeouts thrown": "Strikeouts",
    "trikeouts thrown": "Strikeouts", "irkeouts": "Strikeouts",
    "rkeouts": "Strikeouts", "trikeouts": "Strikeouts",
    "serkeoute twrown": "Strikeouts", "serkeoute": "Strikeouts",
    "strkeoute throw": "Strikeouts",  "strkeoute": "Strikeouts",
    "erkeoute": "Strikeouts",         "trrown": "Strikeouts",
    "rune + r": "Hits+Runs+RBI",
    "hits + runs + rbis": "Hits+Runs+RBI", "hits + runs + rbi": "Hits+Runs+RBI",
    "hits + runs": "Hits+Runs+RBI", "runs + rbis": "Hits+Runs+RBI",
    "home runs": "Home Runs", "home run": "Home Runs",
    "total bases (from hits)": "Total Bases", "total bases from hits": "Total Bases",
    "total bases": "Total Bases", "tal bases": "Total Bases",
    "tal bases (from hs)": "Total Bases", "tal bases (from hits)": "Total Bases",
    "hits": "Hits", "rbis": "RBI", "rbi": "RBI",
    "runs scored": "Runs", "runs": "Runs",
}

COL1 = """
Jy Yamamoto SP (n)
cle @ Lap
stants 52:21
= 65 +
irkeouts Thrown,
More Less

Jy S.Ohtant OF
CLE @ LAD Williams (R)
Starts 52:21
6 2.5 +
te Runs Rls
More

Ih W Bens oF7ss
CLE @ LAD Williams (R)
Starts 52:21
1.5 +
te Runs Rls
More Less

PS a} E
CLE @ LAD Williams (R)
Starts 52:21
1.5 +
Total Bases (From Hits)
"""

COL2 = """
ps Ryan SP
MIN @ KC
stars in 1221
5.5 + =
rikecuts Thrown
More

ce @ LAD
stars n5221
5.5
tikes Tron
More

AY forte
MIN @ KC Cameron (L)
Starts in 12 21
1.5 +
tt Runs Rs
More

MIN @ KC Ryan
Stars in 12:21
1.5
Hits Runs B
More

ky FeFreemman
CLE @ LAD Yamamoto
Stars in 5221
1.5 +
tt Runs Rs
More

CLE @ LAD
Stars in 5221
1.5
Total Bases (From Hits)
More

a haart
MIN @ KC Cameron
Starts in 12 21
0.5 +
Singles
More

WK Tucker oF
CLE @ LAD William
Stars in 5221
0.5 +
Runs Batted In
"""

COL3 = """
GIN. Cameron SP
MIN @ KC
stants in 12:21
+ 45 +
Strikeouts Thrown
More Less

ps Ryan (R)
MIN @ KC
Stars in 12:21
+ 1.5 +
Total Bases From Hits
More Less

pA Kevan OF
CLE @ LAD Yamamoto (R)
Starts 52:21
+ 1.5 +
ts te Runs Rls
More

ps M. Wallner OF
MIN @ KC Cameron (L)
Starts in 12:21
+ 0.5 +
Runs Batted In
"""


_DK_PROP_MAP = {
    "strikeouts thrown": "Strikeouts", "strikeouts": "Strikeouts",
    "irkeouts thrown": "Strikeouts", "rkeouts thrown": "Strikeouts",
    "trikeouts thrown": "Strikeouts", "srkeouts": "Strikeouts",
    "rikecuts thrown": "Strikeouts", "rikecuts": "Strikeouts",
    "tikes tron": "Strikeouts",
    "rune + r": "Hits+Runs+RBI",
    "runs rls": "Hits+Runs+RBI",    # 'te Runs Rls' after garble-prefix strip
    "runs rs": "Hits+Runs+RBI",     # 'tt Runs Rs' after garble-prefix strip
    "hits + runs + rbis": "Hits+Runs+RBI", "hits + runs + rbi": "Hits+Runs+RBI",
    "hits + runs": "Hits+Runs+RBI", "runs + rbis": "Hits+Runs+RBI",
    "tt runs rs": "Hits+Runs+RBI", "tt runs": "Hits+Runs+RBI",
    "te runs rls": "Hits+Runs+RBI", "hits runs b": "Hits+Runs+RBI",
    "ts te runs rls": "Hits+Runs+RBI",
    "home runs": "Home Runs", "home run": "Home Runs",
    "total bases (from hits)": "Total Bases", "total bases from hits": "Total Bases",
    "total bases": "Total Bases", "tal bases": "Total Bases",
    "singles": "Singles",
    "runs batted in": "RBI", "rbis": "RBI", "rbi": "RBI",
    "hits": "Hits", "runs scored": "Runs", "runs": "Runs",
}

_SKIP_WORDS = frozenset({"pm","am","today","bot","top","current","curent",
                          "starts","start","stants","inning","more","less","locked",
                          "srkeouts","irkeouts","strikeouts","rikecuts","singles",
                          "runs","hits","home","total","bases"})

_ls       = re.compile(r'\b(\d{1,2}\.\d|\d{1,2})\b')
_time_pat = re.compile(r'\b\d{1,2}:\d{2}\b')

_POSITIONS = r"(?:SP|RP|OF|1B|2B|3B|SS|C|DH|CF|LF|RF)"
name_pat = re.compile(
    r'([A-Z])\.\s{0,3}([A-Z][A-Za-z\u00C0-\u017E]{1,20}(?:[\s\-][A-Z][A-Za-z]{1,15})?)'
    r'(?:\s+' + _POSITIONS + r')?',
)
fallback_name_pat = re.compile(
    r'(?<![A-Za-z])([A-Z][a-z]{2,15}(?:\s[A-Z][a-z]{2,10})?)'
    r'\s+(?:SP|RP|OF|1B|2B|3B|SS|C\b|DH|CF|LF|RF)',
    re.IGNORECASE,
)
fallback_num_pat   = re.compile(r'(?<![A-Za-z\.])([A-Z][a-z]{3,15})\s+\d{2,3}\b')
ampersand_name_pat = re.compile(r'&\s+([A-Z][a-z]{3,15})\b')


def _extract_line(seg):
    if '@' in seg: return None
    sl = seg.lower()
    if any(w in sl for w in _SKIP_WORDS): return None
    seg = _time_pat.sub('', seg)
    nums = [float(x) for x in _ls.findall(seg) if 0.5 <= float(x) <= 20.0]
    for raw in re.findall(r'\b([1-9][05])\b', seg):
        v = int(raw) / 10
        if 0.5 <= v <= 9.5 and v not in nums:
            nums.append(v)
    if not nums: return None
    frac = [v for v in nums if v != int(v)]
    return frac[0] if frac else nums[0]


def _find_all_names(line):
    results = []
    for m in name_pat.finditer(line):
        fi = m.group(1).upper()
        ln = _POS_STRIP.sub('', m.group(2).strip()).strip()
        if len(ln) >= 2 and not ln.isupper():
            results.append((fi, ln, m.start(), m.end()))
    for m in fallback_name_pat.finditer(line):
        parts = m.group(1).strip().split(); ln = parts[-1]
        if len(ln) >= 2:
            s, e = m.start(), m.end()
            if not any(s < r[3] and e > r[2] for r in results):
                results.append(('?', ln, s, e))
    for m in fallback_num_pat.finditer(line):
        ln = m.group(1).strip()
        if len(ln) >= 4 and ln.lower() not in _SKIP_WORDS:
            s, e = m.start(), m.end()
            if not any(s < r[3] and e > r[2] for r in results):
                results.append(('?', ln, s, e))
    for m in ampersand_name_pat.finditer(line):
        ln = m.group(1).strip(); s, e = m.start(), m.end()
        if not any(s < r[3] and e > r[2] for r in results):
            results.append(('?', ln, s, e))
    if not results:
        lone = re.fullmatch(r'([A-Z][a-z]{3,15})', line.strip())
        if lone and lone.group(1).lower() in _OCR_ALIAS:
            results.append(('?', lone.group(1), 0, len(lone.group(1))))
    results.sort(key=lambda r: r[2])
    return [(r[0], r[1]) for r in results]

sorted_props = sorted(_DK_PROP_MAP.items(), key=lambda kv: -len(kv[0]))

picks = []
seen = set()

for col_text in [COL1, COL2, COL3]:
    lines = [_preprocess(ln.strip()) for ln in col_text.split("\n") if ln.strip()]
    i = 0
    while i < len(lines):
        name_hits = _find_all_names(lines[i])
        if not name_hits:
            i += 1
            continue
        for idx, (first_initial, last_name) in enumerate(name_hits):
            pick = {"display_name": f"{first_initial}. {last_name}" if first_initial != "?" else last_name,
                    "first_initial": first_initial, "last_name": last_name,
                    "line": None, "prop": None}
            for j in range(i + 1, min(i + 15, len(lines))):
                seg = lines[j]
                if pick["line"] is None:
                    ex = _extract_line(seg)
                    if ex is not None:
                        pick["line"] = ex; continue
                if pick["line"] is not None and pick["prop"] is None:
                    cands = [seg.lower()]
                    if j + 1 < len(lines):
                        cands.append((seg + " " + lines[j+1]).lower())
                    for cand in cands:
                        # Collect ALL matching props in order; for merged two-card lines
                        # assign the Nth prop to the Nth name (Tucker=0→TB, Alonso=1→HR)
                        matched = []
                        for dk, internal in sorted_props:
                            if dk in cand and internal not in matched:
                                matched.append(internal)
                        if matched:
                            pick["prop"] = matched[min(idx, len(matched) - 1)]
                            break
                if pick["line"] is not None and pick["prop"] is not None:
                    key = f"{first_initial}.{last_name}|{pick['prop']}|{pick['line']}"
                    if key not in seen:
                        seen.add(key); picks.append(pick)
                    break
        i += 1

print(f"\nExtracted {len(picks)} picks:\n")
for p in picks:
    print(f"  {p['display_name']:22s}  line={p['line']!s:5}  prop={p['prop']}")
print()
