from pathlib import Path

import torch
import pygame
from src.game.snake_game import SnakeGame
from src.state.state_representation import get_state
from src.rl.trainer import QTrainer
from src.rl.memory import ReplayMemory
from src.utils.experiment_tracker import ExperimentTracker
from src.rl.deep_qnet import DeepQNet
import random

# Configuración del experimento
LR = 0.001
GAMMA = 0.8
HIDDEN_SIZE1 = 256
HIDDEN_SIZE2 = 256
BATCH_SIZE = 500
MAX_MEMORY = 100_000
NUM_GAMES = 1000
RENDER_SPEED = 100
EPSILON_START = 80
EPSILON_MIN = 5
EPSILON_DECAY = 0.995


class Agent:
    def __init__(self):
        self.model = DeepQNet(
            input_size=11,
            hidden_size1=HIDDEN_SIZE1,
            hidden_size2=HIDDEN_SIZE2,
            output_size=3,
        )
        self.trainer = QTrainer(self.model, lr=LR, gamma=GAMMA)
        self.memory = ReplayMemory(MAX_MEMORY)
        self.epsilon = EPSILON_START
        self.n_games = 0

    def get_action(self, state):
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
        if len(self.memory) > BATCH_SIZE:
            batch = self.memory.sample(BATCH_SIZE)
        else:
            batch = self.memory.sample(len(self.memory))
        self.trainer.train_step(*batch)


def train():
    agent = Agent()
    game = SnakeGame()
    tracker = ExperimentTracker()
    tracker.start(
        {
            "description": "DeepQNet test with 2 hidden layers",
            "lr": LR,
            "gamma": GAMMA,
            "hidden_size_1": HIDDEN_SIZE1,
            "hidden_size_2": HIDDEN_SIZE2,
            "batch_size": BATCH_SIZE,
            "epsilon_decay": EPSILON_DECAY,
            "epsilon_min": EPSILON_MIN,
        }
    )

    while agent.n_games < NUM_GAMES:
        state_old = get_state(game)
        action = agent.get_action(state_old)
        reward, done, score = game.play_step(action)
        game.render_pygame(speed=RENDER_SPEED)

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
            agent.epsilon = max(EPSILON_MIN, agent.epsilon * EPSILON_DECAY)
            agent.train_long_memory()
            print(f"[{agent.n_games}/{NUM_GAMES}] Score: {score}")
            if tracker.log_score(agent.n_games, score):
                tracker.save_model(agent.model)

    tracker.plot_scores()
    tracker.summarize_run()
    pygame.quit()


if __name__ == "__main__":
    train()
