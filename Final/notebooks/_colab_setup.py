# Shared Colab setup — paste into notebooks or %run
import sys
from pathlib import Path

TRACK = "hourly"  # or "15min"

# Clone repo (Colab)
# !git clone https://github.com/Dinith132/steel-industry-energy-consumption.git

# Mount Drive (Colab)
# from google.colab import drive
# drive.mount("/content/drive")

final_root = Path("/content/Final")
if not final_root.exists():
    final_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(final_root))

from src.config import load_config

cfg = load_config(track=TRACK)
print("Track:", cfg["track"], "| experiment:", cfg["experiment_name"])
