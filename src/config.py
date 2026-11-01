"""Central settings. Every script imports from here."""
from pathlib import Path

# Required by the prof: last two digits of ID 2023A7PS0552P -> 52
SEED = 52

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
LOGS_DIR = ROOT / "logs"

DATA_FILE = DATA_DIR / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"