import re
_ls = re.compile(r'\b(\d{1,2}\.\d|\d{1,2})\b')
_SW = frozenset({'pm','am','today','bot','top','current','curent','starts','start','inning','more','less','locked'})

def ex(seg):
    if '@' in seg: return None
    sl = seg.lower()
    if any(w in sl for w in _SW): return None
    nums = [float(m) for m in _ls.findall(seg) if 0.5 <= float(m) <= 20.0]
    if not nums: return None
    frac = [v for v in nums if v != int(v)]
    return frac[0] if frac else nums[0]

cases = [
    ('c (8) NYY @ SEA', None),
    ('7 80S 0@ HOU 1 Bor 2', None),
    ('starts in 58:50', None),
    ('Today, 1070 PM 1', None),
    ('= 7.5 +', 7.5),
    ('= 5.5 +', 5.5),
    ('+ 8 5.5 +', 5.5),
    ('0.5 +', 0.5),
    ('1.5 +', 1.5),
    ('+ 0.5', 0.5),
    ('75 +', 7.5),
]
all_ok = True
for seg, exp in cases:
    got = ex(seg)
    status = "OK" if got == exp else "FAIL"
    if status == "FAIL":
        all_ok = False
    print(status, repr(seg[:40]), "->", got, "(expected", exp, ")")
print("ALL OK" if all_ok else "FAILURES ABOVE")
