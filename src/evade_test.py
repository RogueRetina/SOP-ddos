"""Step 6: how fragile is the detector? (sensitivity test, not a real attack)

Take the attack flows in the held-out test set and change ONE field,
Init_Win_bytes_forward (the client's initial TCP window), to other values.
Everything else stays the same. Then see how much recall drops.

Caveat: a real attacker who changed their TCP stack would probably change other
features too. This only shows how much the model leans on this one field.

Run:  python src/evade_test.py --n 10
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from config import DATA_FILE, LOGS_DIR, SEED
from prepare import FEATURES

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=10)
parser.add_argument("--depth", type=int, default=6)
args = parser.parse_args()

early4 = ["Destination Port", "Init_Win_bytes_forward",
          "Init_Win_bytes_backward", "min_seg_size_forward"]
sets = {"early4": early4, "all21": FEATURES}

df = pd.read_csv(DATA_FILE)
df.columns = df.columns.str.strip()
df = df.replace([np.inf, -np.inf], np.nan).dropna().drop_duplicates()
df["npkts"] = df["Total Fwd Packets"] + df["Total Backward Packets"]
df["y"] = (df["Label"] != "BENIGN").astype(int)
sub = df[df["npkts"] <= args.n]

train, test = train_test_split(sub, test_size=0.3, stratify=sub["y"], random_state=SEED)
attacks = test[test.y == 1]

models = {
    name: DecisionTreeClassifier(max_depth=args.depth, random_state=SEED)
    .fit(train[f], train["y"])
    for name, f in sets.items()
}

lines = [f"recall on {len(attacks):,} test attack flows after changing "
         f"Init_Win_bytes_forward (depth {args.depth}, flows <= {args.n} packets)",
         f"{'window value':<22}" + "".join(f"{n:>10}" for n in sets)]


def recall(name, data):
    return models[name].predict(data[sets[name]]).mean()   # all rows are attacks


lines.append(f"{'unchanged':<22}" + "".join(f"{recall(n, attacks):>10.4f}" for n in sets))
for v, note in [(229, "229"), (1024, "1024"), (8192, "8192 (Windows-like)"),
                (29200, "29200 (Linux-like)"), (64240, "64240 (modern Linux)")]:
    changed = attacks.copy()
    changed["Init_Win_bytes_forward"] = v
    lines.append(f"{note:<22}" + "".join(f"{recall(n, changed):>10.4f}" for n in sets))

text = "\n".join(lines)
print(text)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
(LOGS_DIR / "step6_evade_test.txt").write_text(text, encoding="utf-8")