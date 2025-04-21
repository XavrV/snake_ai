import os
from pathlib import Path

# Crear estructura base para experimentos
base_dir = Path("/home/javs/Documents/Projects/snake_ai_new/src/data")
subdirs = [
    "runs",
    "runs/run_001",  # ejemplo de primer experimento
    "runs/run_001/frames",  # si se graban juegos
    "src/utils",
]

# Crear directorios si no existen
for subdir in subdirs:
    os.makedirs(base_dir / subdir, exist_ok=True)

# Crear archivos base vacíos o con estructura mínima
files_to_create = {
    base_dir
    / "run_log.csv": "run_id,description,lr,gamma,hidden_size,batch_size,max_score,avg_score,num_games\n",
    base_dir
    / "runs/run_001/config.json": '{\n  "lr": 0.001,\n  "gamma": 0.9,\n  "hidden_size": 256,\n  "batch_size": 1000,\n  "description": "Baseline run"\n}',
    base_dir / "runs/run_001/results.csv": "game,score\n",
}

for path, content in files_to_create.items():
    with open(path, "w") as f:
        f.write(content)

base_dir
