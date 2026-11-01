"""Step 4: understand WHY the detector works and WHERE it fails.

Part 1: how do the two big features (Fwd Packet Length Max, URG Flag Count)
        differ between benign and attack flows?
Part 2: the missed attacks (false negatives) of the depth-6 tree: what do
        they look like compared with caught attacks and with benign port-80
        traffic?

Run:  python src/analyze.py        (saves logs/step4_analysis.txt)
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from config import LOGS_DIR, RESULTS_DIR, SEED
from prepare import load_clean

LOGS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
lines = []


def say(t=""):
    print(t)
    lines.append(str(t))


X, y = load_clean()
label = y.map({0: "benign", 1: "attack"})

# ---------------- Part 1: the two big features -------------------------
say("=== PART 1: key features by class (all cleaned flows) ===")
say("\nURG Flag Count: share of flows with each value")
urg = pd.crosstab(X["URG Flag Count"], label, normalize="columns").round(4)
say(urg.head(10).to_string())
if len(urg) > 10:
    say(f"... ({len(urg)} distinct values, showing first 10)")

say("\nFwd Packet Length Max: quantiles")
q = X["Fwd Packet Length Max"].groupby(label).quantile([.1, .5, .9]).unstack()
say(q.round(1).to_string())

# ---------------- Part 2: false negatives ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=SEED)
tree = DecisionTreeClassifier(max_depth=6, random_state=SEED).fit(X_train, y_train)
pred = tree.predict(X_test)

t = X_test.copy()
t["y"], t["pred"] = y_test, pred
missed = t[(t.y == 1) & (t.pred == 0)]
caught = t[(t.y == 1) & (t.pred == 1)]
benign80 = t[(t.y == 0) & (t["Destination Port"] == 80)]

say("\n=== PART 2: the missed attacks (depth 6, all features, test set) ===")
say(f"missed attacks: {len(missed)}   caught attacks: {len(caught):,}   "
    f"benign flows on port 80: {len(benign80):,}")
say(f"ports of missed attacks: {missed['Destination Port'].value_counts().head(5).to_dict()}")

top = pd.Series(tree.feature_importances_, index=X.columns).nlargest(8).index
table = pd.DataFrame({
    "missed attacks": missed[top].median(),
    "caught attacks": caught[top].median(),
    "benign port 80": benign80[top].median(),
})
say("\nmedian feature values (top-8 features by importance):")
say(table.round(2).to_string())

missed.to_csv(RESULTS_DIR / "false_negatives_depth6_all_full.csv", index=False)
(LOGS_DIR / "step4_analysis.txt").write_text("\n".join(lines), encoding="utf-8")
say(f"\n(saved to {LOGS_DIR / 'step4_analysis.txt'})")