"""Step 5: make the 'first few packets' constraint honest.

Problem: the CIC-IDS2017 CSV has features computed over the WHOLE flow.
Fix: keep only flows with at most N packets in total. For those flows the
first N packets ARE the whole flow, so the CSV values are exactly what a
switch would have after seeing N packets. (Packet counts are used only to
select flows for this experiment, never as a model feature.)

We compare four feature sets on that subset:
  early4          port + initial TCP windows + min segment size
                  (known from the first 1-2 packets)
  early4_noport   the same without the destination port
  all21           the 21 switch-style features from prepare.py
  all21_noport    the same without the destination port

Run:  python src/first_n.py --n 10
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from config import DATA_FILE, LOGS_DIR, SEED
from prepare import FEATURES

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=10, help="max packets per flow")
parser.add_argument("--depth", type=int, default=6)
args = parser.parse_args()

LOGS_DIR.mkdir(parents=True, exist_ok=True)
lines = []


def say(t=""):
    print(t)
    lines.append(str(t))


# ---- load + clean (same cleaning as prepare.load_clean) ------------------
df = pd.read_csv(DATA_FILE)
df.columns = df.columns.str.strip()
df = df.replace([np.inf, -np.inf], np.nan).dropna().drop_duplicates()
df = df.reset_index(drop=True)
df["npkts"] = df["Total Fwd Packets"] + df["Total Backward Packets"]
df["y"] = (df["Label"] != "BENIGN").astype(int)

say("=== how many packets do flows have? ===")
say(df.groupby("y")["npkts"].describe(percentiles=[.5, .9, .99])
    .round(1).rename(index={0: "benign", 1: "attack"}).to_string())
say("\nshare of flows with at most k packets (benign / attack):")
for k in (3, 5, 10, 20, 50):
    b = (df.loc[df.y == 0, "npkts"] <= k).mean()
    a = (df.loc[df.y == 1, "npkts"] <= k).mean()
    say(f"  k={k:<3}  benign {b:6.1%}   attack {a:6.1%}")

# ---- subset ---------------------------------------------------------------
sub = df[df["npkts"] <= args.n]
say(f"\n=== subset: flows with at most {args.n} packets ===")
say(f"flows kept: {len(sub):,} of {len(df):,} ({len(sub) / len(df):.1%})")
say(f"benign: {(sub.y == 0).sum():,}   attack: {(sub.y == 1).sum():,}")
if sub.y.nunique() < 2 or sub.y.value_counts().min() < 50:
    raise SystemExit("Too few flows of one class in this subset; try a larger --n")

early4 = ["Destination Port", "Init_Win_bytes_forward",
          "Init_Win_bytes_backward", "min_seg_size_forward"]
sets = {
    "early4": early4,
    "early4_noport": [f for f in early4 if f != "Destination Port"],
    "all21": FEATURES,
    "all21_noport": [f for f in FEATURES if f != "Destination Port"],
}

train, test = train_test_split(sub, test_size=0.3, stratify=sub["y"], random_state=SEED)

say(f"\n=== results on the held-out test split (seed {SEED}, depth {args.depth}) ===")
say(f"{'feature set':<15}{'precision':>10}{'recall':>9}{'f1':>8}   TN / FP / FN / TP")
for name, feats in sets.items():
    tree = DecisionTreeClassifier(max_depth=args.depth, random_state=SEED)
    tree.fit(train[feats], train["y"])
    pred = tree.predict(test[feats])
    tn, fp, fn, tp = confusion_matrix(test["y"], pred).ravel()
    say(f"{name:<15}{precision_score(test['y'], pred):>10.4f}"
        f"{recall_score(test['y'], pred):>9.4f}{f1_score(test['y'], pred):>8.4f}"
        f"   {tn:,} / {fp:,} / {fn:,} / {tp:,}")

(LOGS_DIR / f"step5_first{args.n}.txt").write_text("\n".join(lines), encoding="utf-8")
say(f"\n(saved to {LOGS_DIR / f'step5_first{args.n}.txt'})")