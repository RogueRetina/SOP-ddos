"""Step 3: train a shallow decision tree and report precision / recall.

Usage (from project root):
    python src/train.py                                  # all 21 features
    python src/train.py --drop "Destination Port"        # honest run, no port shortcut
    python src/train.py --depth 4                        # change tree depth

Each run saves its outputs under results/ and logs/ with a tag in the name.
Attack = positive class (1).
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")                      # no window needed, just save PNGs
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix, precision_score, recall_score,
                             f1_score)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

from config import LOGS_DIR, RESULTS_DIR, SEED
from prepare import FEATURES, load_clean

parser = argparse.ArgumentParser()
parser.add_argument("--depth", type=int, default=6, help="max tree depth")
parser.add_argument("--drop", nargs="*", default=[],
                    help='features to remove, e.g. --drop "Destination Port"')
args = parser.parse_args()

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

features = [f for f in FEATURES if f not in args.drop]
tag = f"depth{args.depth}" + ("_no-" + "-".join(
    d.replace(" ", "") for d in args.drop) if args.drop else "_all")

lines = []


def say(t=""):
    print(t)
    lines.append(t)


# --- data + split -------------------------------------------------------
X, y = load_clean()
X = X[features]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=SEED)

say(f"run tag      : {tag}")
say(f"seed         : {SEED}")
say(f"features ({len(features)}): dropped = {args.drop or 'none'}")
say(f"train / test : {len(X_train):,} / {len(X_test):,}")

# --- train --------------------------------------------------------------
tree = DecisionTreeClassifier(max_depth=args.depth, random_state=SEED)
tree.fit(X_train, y_train)
pred = tree.predict(X_test)

# --- metrics ------------------------------------------------------------
m = {
    "precision": precision_score(y_test, pred),
    "recall": recall_score(y_test, pred),
    "f1": f1_score(y_test, pred),
    "accuracy": accuracy_score(y_test, pred),
}
tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
say("\n--- test-set results (attack = positive) ---")
say(f"precision : {m['precision']:.4f}   (of flagged flows, how many were real attacks)")
say(f"recall    : {m['recall']:.4f}   (of real attacks, how many we caught)")
say(f"f1        : {m['f1']:.4f}")
say(f"accuracy  : {m['accuracy']:.4f}   (for reference only)")
say(f"TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}")

# --- feature importances -----------------------------------------------
imp = pd.Series(tree.feature_importances_, index=features).sort_values(ascending=False)
say("\n--- top 8 feature importances ---")
for f, v in imp.head(8).items():
    say(f"  {f:<26} {v:.3f}")

# --- where does it fail? -----------------------------------------------
say("\n--- errors by destination port (only if port was a feature) ---")
test = X_test.copy()
test["y"] = y_test
test["pred"] = pred
if "Destination Port" in test:
    fp_rows = test[(test.y == 0) & (test.pred == 1)]
    fn_rows = test[(test.y == 1) & (test.pred == 0)]
    say(f"false positives (benign flagged as attack): {len(fp_rows):,}")
    say(f"  top ports: {fp_rows['Destination Port'].value_counts().head(3).to_dict()}")
    say(f"false negatives (attack missed): {len(fn_rows):,}")
    say(f"  top ports: {fn_rows['Destination Port'].value_counts().head(3).to_dict()}")
else:
    say("(port not used; inspect false negatives manually, see results file)")
    test[(test.y == 1) & (test.pred == 0)].head(20).to_csv(
        RESULTS_DIR / f"false_negatives_{tag}.csv", index=False)

# --- save everything ----------------------------------------------------
(RESULTS_DIR / f"metrics_{tag}.json").write_text(
    json.dumps({**m, "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
                "seed": SEED, "depth": args.depth, "dropped": args.drop}, indent=2))
(RESULTS_DIR / f"tree_rules_{tag}.txt").write_text(
    export_text(tree, feature_names=features))

fig, ax = plt.subplots(figsize=(4.5, 4))
ConfusionMatrixDisplay.from_predictions(
    y_test, pred, display_labels=["benign", "attack"], ax=ax, colorbar=False)
ax.set_title(f"Confusion matrix ({tag})")
fig.tight_layout()
fig.savefig(RESULTS_DIR / f"confusion_{tag}.png", dpi=150)

(LOGS_DIR / f"step3_{tag}.txt").write_text("\n".join(lines), encoding="utf-8")
say(f"\nsaved: results/metrics_{tag}.json, confusion_{tag}.png, "
    f"tree_rules_{tag}.txt, logs/step3_{tag}.txt")