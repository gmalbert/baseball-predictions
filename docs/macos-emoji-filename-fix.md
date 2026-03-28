# Fix: Streamlit Emoji Filenames Failing on macOS / Docker

## Status
- [x] Complete

## Problem

Streamlit page files with emoji in their names (e.g. `4_💰_Value_Finder.py`) cause a
`FileNotFoundError` on macOS and Linux/Docker, even though the same files work on Windows.

```
FileNotFoundError: [Errno 2] No such file or directory: '/app/pages/4_??_Value_Finder.py'
```

**Root cause:** Windows uses NFC Unicode normalization for emoji in filenames; macOS uses NFD.
When a repo is cloned on macOS (or run in a Linux Docker container), the emoji characters in
filenames are decoded differently, so Streamlit cannot resolve the path it stores internally.

---

## Solution Overview

1. Rename all page files to remove emoji from filenames.
2. Use `st.navigation()` in the entry point to declare icons explicitly — the sidebar still
   shows emoji even though the filenames are plain ASCII.
3. Remove `st.set_page_config()` from every sub-page (required by `st.navigation()`; config
   must be called exactly once, in the entry point).
4. Update any `st.switch_page()` calls to use the new filenames.

---

## Step-by-Step Instructions

### 1. Rename page files (remove emoji)

Use `git mv` to preserve history. Run from the repo root:

```bash
cd pages/
git mv "1_📅_Todays_Games.py"      "1_Todays_Games.py"
git mv "2_📊_Team_Stats.py"         "2_Team_Stats.py"
git mv "3_🏆_Standings.py"          "3_Standings.py"
git mv "4_💰_Value_Finder.py"       "4_Value_Finder.py"
git mv "5_🎯_Player_Props.py"       "5_Player_Props.py"
git mv "6_📈_Performance.py"        "6_Performance.py"
git mv "7_🥅_Goalies.py"            "7_Goalies.py"
git mv "8_🏥_Injuries.py"           "8_Injuries.py"
git mv "9_📉_Line_Movement.py"      "9_Line_Movement.py"
git mv "10_📊_Model_Performance.py" "10_Model_Performance.py"
git mv "11_🔬_Backtesting.py"       "11_Backtesting.py"
```

Adapt names to match whatever pages exist in the target repo.

---

### 2. Remove `st.set_page_config()` from every page file

**⚠️ CRITICAL — do this carefully. The PowerShell one-liner below is the safe way.**

When using `st.navigation()`, `set_page_config` can only be called once (in the entry point).
Each sub-page must have its `set_page_config(...)` line deleted.

Run this from the repo root (PowerShell — works on Windows and macOS `pwsh`):

```powershell
$base = "./pages"
Get-ChildItem "$base/*.py" | ForEach-Object {
    $content = [System.IO.File]::ReadAllText($_.FullName, [System.Text.Encoding]::UTF8)
    $updated = $content -replace '(?m)^st\.set_page_config\([^\r\n]*\)\r?\n', ''
    [System.IO.File]::WriteAllText($_.FullName, $updated, [System.Text.Encoding]::UTF8)
    Write-Host "Updated: $($_.Name)"
}
```

Or with bash (macOS/Linux):

```bash
for f in pages/*.py; do
    sed -i '' '/^st\.set_page_config(/d' "$f"
    echo "Updated: $f"
done
```

**Verify no calls remain:**

```bash
grep -r "set_page_config" pages/
# Should return nothing
```

---

### 3. Rewrite the entry point (`predictions.py` / `app.py`)

Replace the top-level page content with a `home_page()` function, add `st.navigation()` at
the bottom, and keep `st.set_page_config()` called exactly once at the top level.

**Template structure:**

```python
"""Entry point."""
import streamlit as st
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
# ... your other imports ...

# Called ONCE here — sub-pages must NOT call set_page_config
st.set_page_config(
    page_title="Your App Title",
    page_icon="🏒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ... any app-wide setup (CSS, cache_resource, etc.) ...


def home_page():
    """Landing page content."""
    # All your existing top-level landing page code goes here,
    # indented into this function.
    st.title("...")
    # ...
    # Update any st.switch_page() calls to new filenames (no emoji)
    if st.button("Go somewhere"):
        st.switch_page("pages/1_Todays_Games.py")   # no emoji in path


# Navigation — icons are declared here, not in filenames
pg = st.navigation(
    {
        "": [
            st.Page(home_page, title="Home", icon="🏒", default=True),
        ],
        "Analytics": [
            st.Page("pages/1_Todays_Games.py",      title="Today's Games",    icon="📅"),
            st.Page("pages/2_Team_Stats.py",         title="Team Stats",       icon="📊"),
            st.Page("pages/3_Standings.py",          title="Standings",        icon="🏆"),
            st.Page("pages/4_Value_Finder.py",       title="Value Finder",     icon="💰"),
            st.Page("pages/5_Player_Props.py",       title="Player Props",     icon="🎯"),
            st.Page("pages/6_Performance.py",        title="Performance",      icon="📈"),
            st.Page("pages/7_Goalies.py",            title="Goalies",          icon="🥅"),
            st.Page("pages/8_Injuries.py",           title="Injuries",         icon="🏥"),
            st.Page("pages/9_Line_Movement.py",      title="Line Movement",    icon="📉"),
        ],
        "Models": [
            st.Page("pages/10_Model_Performance.py", title="Model Performance", icon="📊"),
            st.Page("pages/11_Backtesting.py",       title="Backtesting",       icon="🔬"),
        ],
    }
)
pg.run()
```

---

### 4. Update any remaining `st.switch_page()` calls

Search the whole project for old emoji-based paths:

```bash
grep -r "switch_page" . --include="*.py"
```

Replace every occurrence like:
```python
# Before
st.switch_page("pages/4_💰_Value_Finder.py")

# After
st.switch_page("pages/4_Value_Finder.py")
```

---

## Commit

```bash
git add -A
git commit -m "fix: remove emoji from page filenames for macOS/Docker compatibility

- Renamed pages/N_EMOJI_Name.py -> pages/N_Name.py
- Added st.navigation() with explicit icons to entry point
- Removed st.set_page_config() from all sub-pages
- Updated all st.switch_page() references"
git push
```

---

## ⚠️ Lessons Learned / Pitfalls

### The PowerShell working-directory trap
The `set_page_config` removal script **must use absolute paths** (or `Join-Path` with a
known base). If the shell's working directory was changed by a prior command that failed
silently, `[System.IO.File]::ReadAllText("filename.py")` resolves relative to the *wrong*
directory — it will silently read the wrong file (or fail), and then `WriteAllText` will
**overwrite the target file with the wrong content**. Every page ended up with backtesting
content because the script read the wrong file.

**Safe pattern (always use absolute paths):**
```powershell
$base = "C:\absolute\path\to\repo\pages"   # always absolute
Get-ChildItem "$base/*.py" | ForEach-Object {
    # $_.FullName is always absolute — safe
    $content = [System.IO.File]::ReadAllText($_.FullName, [System.Text.Encoding]::UTF8)
    ...
    [System.IO.File]::WriteAllText($_.FullName, $updated, [System.Text.Encoding]::UTF8)
}
```

### Verify file contents before committing
After any bulk file operation, spot-check a few files before `git add`:
```bash
head -3 pages/1_Todays_Games.py
head -3 pages/4_Value_Finder.py
head -3 pages/11_Backtesting.py
```
If they all show the same content, something went wrong — do **not** commit.

### Recovery when pages are corrupted
If you accidentally commit corrupted page content, recover from the last good commit:
```bash
# Find the last good commit hash
git log --oneline -5

# Restore the pages/ directory from that commit
git checkout <good-commit-hash> -- pages/

# Redo the set_page_config removal correctly, then commit
```

### `git mv` with emoji on Windows
`git mv "emoji_file.py" "plain_file.py"` works on Windows but git's similarity detection
may show confusing rename pairings in `git status`. The actual files on disk are correct —
verify with `Get-ChildItem pages/` rather than trusting the `git status` display.

### `st.navigation()` requires removing `set_page_config` from sub-pages
Streamlit raises `StreamlitSetPageConfigMustBeFirstCommandError` if any sub-page calls
`set_page_config`. This is safe to remove because `st.navigation()` inherits the config
(title, icon, layout) set in the entry point.
