"""Which of the 4 early features does the tree use, and how?

Run:  python src/check_early4.py --n 10
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

from config import DATA_FILE, LOGS_DIR, SEED

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=10)
args = parser.parse_args()

early4 = ["Destination Port", "Init_Win_bytes_forward",
          "Init_Win_bytes_backward", "min_seg_size_forward"]

df = pd.read_csv(DATA_FILE)
df.columns = df.columns.str.strip()
df = df.replace([np.inf, -np.inf], np.nan).dropna().drop_duplicates()
df["npkts"] = df["Total Fwd Packets"] + df["Total Backward Packets"]
df["y"] = (df["Label"] != "BENIGN").astype(int)
sub = df[df["npkts"] <= args.n]

train, test = train_test_split(sub, test_size=0.3, stratify=sub["y"], random_state=SEED)
tree = DecisionTreeClassifier(max_depth=3, random_state=SEED).fit(train[early4], train["y"])

out = []
out.append(f"=== depth-3 tree on the 4 early features (flows with <= {args.n} packets) ===")
out.append(export_text(tree, feature_names=early4))
out.append("importances: " + str(dict(zip(early4, tree.feature_importances_.round(3)))))
for col in ("Init_Win_bytes_backward", "min_seg_size_forward"):
    out.append(f"\ntop values of {col} per class:")
    out.append(sub.groupby("Label")[col].value_counts().groupby(level=0).head(3).to_string())

text = "\n".join(out)
print(text)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
(LOGS_DIR / "step5b_early4_check.txt").write_text(text, encoding="utf-8")