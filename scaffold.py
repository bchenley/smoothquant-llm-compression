import os
from pathlib import Path

folders = [
    "data/sample",
    "src",
    "notebooks",
    "results/logs",
    "results/plots",
    "reports/summary",
    "reports/slides",
    "reports/figures",
    "scripts"
]

for folder in folders:
    path = Path(folder)
    path.mkdir(parents=True, exist_ok=True)
    gitkeep = path / ".gitkeep"
    gitkeep.touch(exist_ok=True)

print("Project folders created successfully.")
