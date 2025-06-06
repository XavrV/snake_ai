import torch
import pygame
from itertools import product
from src.game.snake_game import SnakeGame
from src.state.state_representation import get_state
from src.rl.linear_qnet import LinearQNet
from src.rl.trainer import QTrainer
from src.rl.memory import ReplayMemory
from src.utils.experiment_tracker import ExperimentTracker
import random
import pandas as pd

# Experimentos con combinaciones múltiples
learning_rates = [0.001]
gammas = [0.8, 0.85]
hidden_sizes = [512, 1024]
batch_sizes = [500]

NUM_GAMES = 1000
MAX_MEMORY = 100_000
RENDER_SPEED = 100  # +o- 20


class Agent:
    def __init__(self, lr, gamma, hidden_size, batch_size):
        self.model = LinearQNet(input_size=11, hidden_size=hidden_size, output_size=3)
        self.trainer = QTrainer(self.model, lr=lr, gamma=gamma)
        self.memory = ReplayMemory(MAX_MEMORY)
        self.batch_size = batch_size
        self.epsilon = 0
        self.n_games = 0

    def get_action(self, state):
        self.epsilon = max(5, 80 - 0.995 * self.n_games)
        # self.epsilon = 80 - self.n_games
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            action = [0, 0, 0]
            action[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = int(torch.argmax(prediction))
            action = [0, 0, 0]
            action[move] = 1
        return action

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def remember(self, state, action, reward, next_state, done):
        self.memory.push((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > self.batch_size:
            batch = self.memory.sample(self.batch_size)
        else:
            batch = self.memory.sample(len(self.memory))
        self.trainer.train_step(*batch)


def run_experiment(lr, gamma, hidden_size, batch_size, results_summary):
    agent = Agent(lr, gamma, hidden_size, batch_size)
    game = SnakeGame()
    tracker = ExperimentTracker()
    tracker.start(
        {
            "description": f"LR={lr}, gamma={gamma}, hidden={hidden_size}, batch={batch_size}",
            "lr": lr,
            "gamma": gamma,
            "hidden_size": hidden_size,
            "batch_size": batch_size,
        }
    )

    while agent.n_games < NUM_GAMES:
        state_old = get_state(game)
        action = agent.get_action(state_old)
        reward, done, score = game.play_step(action)
        # game.render_pygame(speed=RENDER_SPEED)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                tracker.plot_scores()
                tracker.summarize_run()
                exit()

        state_new = get_state(game)
        agent.train_short_memory(state_old, action, reward, state_new, done)
        agent.remember(state_old, action, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            print(
                f"[{agent.n_games}/{NUM_GAMES}] LR={lr} | γ={gamma} | h={hidden_size} | batch={batch_size} | Score: {score}"
            )
            if tracker.log_score(agent.n_games, score):
                tracker.save_model(agent.model)

    tracker.plot_scores()
    tracker.summarize_run()
    results_summary.append(
        {
            "lr": lr,
            "gamma": gamma,
            "hidden_size": hidden_size,
            "batch_size": batch_size,
            "max_score": tracker.best_score,
            "avg_score": sum(s for _, s in tracker.scores) / len(tracker.scores),
        }
    )
    pygame.quit()


if __name__ == "__main__":
    all_results = []
    combinations = list(product(learning_rates, gammas, hidden_sizes, batch_sizes))
    for lr, gamma, h, b in combinations:
        run_experiment(lr, gamma, h, b, all_results)

    # Mostrar resumen ordenado por max_score
    df = pd.DataFrame(all_results)
    df = df.sort_values(by="max_score", ascending=False)
    print("\\n=== RESUMEN FINAL ORDENADO POR MAX SCORE ===")
    print(df.to_string(index=False))

    # crear archivo de texto con las metricas en tabla
    with open("experiments_summary.txt", "w") as f:
        f.write(df.to_string(index=False))
    # guardar el dataframe como csv
    df.to_csv("experiments_summary.csv", index=False)
