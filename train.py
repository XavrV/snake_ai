from src.game.snake_game import SnakeGame
from src.state.state_representation import get_state
from src.rl.linear_qnet import LinearQNet
from src.rl.trainer import QTrainer
from src.rl.memory import ReplayMemory

import numpy as np
import torch
import random
import pygame
import matplotlib.pyplot as plt

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:
    def __init__(self):
        self.model = LinearQNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=0.9)
        self.memory = ReplayMemory(MAX_MEMORY)
        self.epsilon = 0
        self.n_games = 0

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            action = [0, 0, 0]
            action[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            action = [0, 0, 0]
            action[move] = 1
        return action

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def remember(self, state, action, reward, next_state, done):
        self.memory.push((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            batch = self.memory.sample(BATCH_SIZE)
        else:
            batch = self.memory.sample(len(self.memory))
        self.trainer.train_step(*batch)


scores = []
mean_scores = []


def plot_final(scores, mean_scores):
    plt.figure(figsize=(10, 5))
    plt.title("Progreso Final del Entrenamiento")
    plt.xlabel("Número de Juegos")
    plt.ylabel("Score")
    plt.plot(scores, label="Score por juego")
    plt.plot(mean_scores, label="Score promedio")
    plt.legend()
    plt.grid(True)
    plt.savefig("training_progress.png")
    plt.show()


def train():
    agent = Agent()
    game = SnakeGame()

    try:
        while True:
            state_old = get_state(game)
            action = agent.get_action(state_old)
            reward, done, score = game.play_step(action)
            game.render_pygame(speed=20)  # opcional
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    raise KeyboardInterrupt()

            state_new = get_state(game)

            agent.train_short_memory(state_old, action, reward, state_new, done)
            agent.remember(state_old, action, reward, state_new, done)

            if done:
                game.reset()
                agent.n_games += 1
                agent.train_long_memory()
                scores.append(score)
                mean_score = sum(scores) / len(scores)
                mean_scores.append(mean_score)
                print(
                    f"Juego {agent.n_games} - Score: {score} - Promedio: {mean_score:.2f}"
                )

    except KeyboardInterrupt:
        print("Entrenamiento interrumpido. Guardando gráfico final...")
        plot_final(scores, mean_scores)


if __name__ == "__main__":
    train()
