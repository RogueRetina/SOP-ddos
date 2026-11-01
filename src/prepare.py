"""Step 2: clean the data and keep ONLY switch-computable features.

The prof's rule: features must be things a network switch could compute from
the first few packets of a flow. We therefore do NOT use anything that needs
the whole flow (duration, total packets/bytes, rates, idle/active times).

Honest caveat (put this in the report): the CIC-IDS2017 CSV only gives
features computed over the WHOLE flow. The features below are ones a switch
could keep as simple running counters / min / max over the first N packets,
but here they are approximated by the full-flow values.

Run:  python src/prepare.py      (prints + logs a summary)
Import: from prepare import load_clean
"""
import numpy as np
import pandas as pd

from config import DATA_FILE, LOGS_DIR

# feature -> why a switch can compute it
FEATURE_REASONS = {
    # known from the very first packets
    "Destination Port": "in the first packet header",
    "Init_Win_bytes_forward": "TCP window in the SYN (first packet)",
    "Init_Win_bytes_backward": "TCP window in the SYN-ACK (second packet)",
    "min_seg_size_forward": "from first packet's TCP header",
    # running min / max / mean of packet sizes (a few registers)
    "Fwd Packet Length Max": "running max over fwd packets",
    "Fwd Packet Length Min": "running min over fwd packets",
    "Fwd Packet Length Mean": "running sum / count over fwd packets",
    "Bwd Packet Length Max": "running max over bwd packets",
    "Bwd Packet Length Min": "running min over bwd packets",
    "Bwd Packet Length Mean": "running sum / count over bwd packets",
    "Min Packet Length": "running min over all packets",
    "Max Packet Length": "running max over all packets",
    # TCP flag counters (one counter per flag)
    "SYN Flag Count": "counter on the SYN flag",
    "ACK Flag Count": "counter on the ACK flag",
    "RST Flag Count": "counter on the RST flag",
    "PSH Flag Count": "counter on the PSH flag",
    "FIN Flag Count": "counter on the FIN flag",
    "URG Flag Count": "counter on the URG flag",
    # gap between packets (needs last-seen timestamp only)
    "Flow IAT Min": "running min of gap between packets",
    "Flow IAT Max": "running max of gap between packets",
    "Flow IAT Mean": "running mean of gap between packets",
}
FEATURES = list(FEATURE_REASONS)


def load_clean(verbose=False):
    """Return (X, y): X = switch-computable features, y = 0 benign / 1 attack."""
    df = pd.read_csv(DATA_FILE)
    df.columns = df.columns.str.strip()          # ' Label' -> 'Label'
    n0 = len(df)

    # inf (division by zero) -> NaN, then drop rows that have any NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    n_bad = int(df.isna().any(axis=1).sum())
    df = df.dropna()

    # drop exact duplicate flows (otherwise same row can be in train AND test)
    n_dup = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)

    if verbose:
        print(f"start rows        : {n0:,}")
        print(f"dropped inf/NaN   : {n_bad:,}")
        print(f"dropped duplicates: {n_dup:,}")
        print(f"final rows        : {len(df):,}")

    y = (df["Label"] != "BENIGN").astype(int)    # 1 = attack
    X = df[FEATURES].copy()
    return X, y


if __name__ == "__main__":
    LOGS_DIR.mkdir(exist_ok=True)
    out = []

    def say(t=""):
        print(t)
        out.append(t)

    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        X, y = load_clean(verbose=True)
    for line in buf.getvalue().splitlines():
        say(line)

    say(f"\nfeatures kept: {len(FEATURES)}")
    for f, why in FEATURE_REASONS.items():
        say(f"  {f:<26} {why}")

    say("\nlabel balance (1 = attack):")
    for k, n in y.value_counts().items():
        say(f"  {k}: {n:,} ({n / len(y):.2%})")

    say("\nattack share by destination port (top 8 by flow count):")
    port = X["Destination Port"]
    top = port.value_counts().head(8).index
    for p in top:
        m = port == p
        say(f"  port {int(p):>5}: {m.sum():>7,} flows, {y[m].mean():6.1%} attack")

    (LOGS_DIR / "step2_prepare.txt").write_text("\n".join(out), encoding="utf-8")
    say(f"\n(saved to {LOGS_DIR / 'step2_prepare.txt'})")