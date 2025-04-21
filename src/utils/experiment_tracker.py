import os
import json
import uuid
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path


class ExperimentTracker:
    def __init__(self, base_dir="experiments"):
        self.base_dir = Path(base_dir)
        self.run_id = (
            f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"
        )
        self.run_path = self.base_dir / "runs" / self.run_id
        self.run_path.mkdir(parents=True, exist_ok=True)
        self.config = {}
        self.scores = []
        self.best_score = 0

    def start(self, config: dict):
        self.config = config
        config_path = self.run_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    def log_score(self, game_num: int, score: int):
        self.scores.append((game_num, score))
        results_path = self.run_path / "results.csv"
        pd.DataFrame(self.scores, columns=["game", "score"]).to_csv(
            results_path, index=False
        )
        if score > self.best_score:
            self.best_score = score
            return True  # nuevo mejor score
        return False

    def save_model(self, model):
        model_path = self.run_path / "model_best.pth"
        model.save(str(model_path))

    def plot_scores(self):
        if not self.scores:
            return
        games, scores = zip(*self.scores)
        plt.figure(figsize=(10, 5))
        plt.plot(games, scores, label="Score por juego")
        # media móvil
        window = 50
        if len(scores) >= window:
            rolling_avg = pd.Series(scores).rolling(window).mean()
            plt.plot(games, rolling_avg, label=f"Media móvil ({window})")
        plt.xlabel("Juego")
        plt.ylabel("Score")
        plt.title("Progreso del agente")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(self.run_path / "plot.png")
        plt.close()

    def summarize_run(self):
        summary_path = self.base_dir / "run_log.csv"
        summary_data = {
            "run_id": self.run_id,
            "description": self.config.get("description", ""),
            "lr": self.config.get("lr"),
            "gamma": self.config.get("gamma"),
            "hidden_size": self.config.get("hidden_size"),
            "batch_size": self.config.get("batch_size"),
            "max_score": self.best_score,
            "avg_score": sum(score for _, score in self.scores) / len(self.scores),
            "num_games": len(self.scores),
        }
        df = pd.DataFrame([summary_data])
        if summary_path.exists():
            df.to_csv(summary_path, mode="a", header=False, index=False)
        else:
            df.to_csv(summary_path, index=False)
