"""Step 1: look at the raw data BEFORE doing any ML.

Answers: how big is it, what are the columns, how many benign vs attack,
and what is dirty (inf / NaN / duplicates / whitespace in column names).
Everything printed is also saved to logs/step1_explore.txt (raw output the
prof asked for).

Run from the project root:
    python src/explore.py
"""
import sys
import numpy as np
import pandas as pd

from config import DATA_FILE, LOGS_DIR

LOGS_DIR.mkdir(exist_ok=True)
log_path = LOGS_DIR / "step1_explore.txt"
log_file = open(log_path, "w", encoding="utf-8")


def say(*args):
    """print to screen AND to the log file"""
    text = " ".join(str(a) for a in args)
    print(text)
    log_file.write(text + "\n")


# --- 1. Load -----------------------------------------------------------
if not DATA_FILE.exists():
    sys.exit(f"CSV not found at {DATA_FILE}. Check the name in src/config.py")

df = pd.read_csv(DATA_FILE)
say("=== 1. SHAPE ===")
say(f"rows: {df.shape[0]:,}   columns: {df.shape[1]}")

# --- 2. Column names ---------------------------------------------------
# CIC-IDS CSVs have leading spaces like ' Destination Port'. Strip them.
raw_cols = list(df.columns)
n_spaced = sum(c != c.strip() for c in raw_cols)
df.columns = df.columns.str.strip()
say("\n=== 2. COLUMNS ===")
say(f"{n_spaced} column names had stray spaces (now stripped)")
for i, c in enumerate(df.columns):
    say(f"{i:3d}  {c}")

# --- 3. Labels ---------------------------------------------------------
say("\n=== 3. LABEL DISTRIBUTION ===")
counts = df["Label"].value_counts()
for label, n in counts.items():
    say(f"{label:<12} {n:>9,}   {n / len(df):6.2%}")

# --- 4. Dirty values ---------------------------------------------------
numeric = df.select_dtypes(include=[np.number])
n_inf = int(np.isinf(numeric).sum().sum())
n_nan = int(df.isna().sum().sum())
say("\n=== 4. DIRTY VALUES ===")
say(f"inf values : {n_inf:,}")
say(f"NaN values : {n_nan:,}")
say(f"duplicate rows: {df.duplicated().sum():,}")

bad_cols = {}
for c in numeric.columns:
    k = int(np.isinf(numeric[c]).sum() + numeric[c].isna().sum())
    if k:
        bad_cols[c] = k
say(f"columns containing inf/NaN: {bad_cols if bad_cols else 'none'}")

# --- 5. Columns that would leak identity -------------------------------
say("\n=== 5. LEAKAGE CHECK ===")
leaky = [c for c in df.columns
         if c.lower() in ("flow id", "source ip", "destination ip",
                          "source port", "timestamp")]
say(f"identity/time columns present: {leaky if leaky else 'none'}")

say(f"\n(saved to {log_path})")
log_file.close()