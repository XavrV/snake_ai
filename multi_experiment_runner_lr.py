import torch
import pygame
from src.game.snake_game import SnakeGame
from src.state.state_representation import get_state
from src.rl.linear_qnet import LinearQNet
from src.rl.trainer import QTrainer
from src.rl.memory import ReplayMemory
from src.utils.experiment_tracker import ExperimentTracker
import random

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
GAMMA = 0.9
HIDDEN_SIZE = 256
NUM_GAMES = 200
RENDER_SPEED = 60  # +o- 20

learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01]


class Agent:
    def __init__(self, lr):
        self.model = LinearQNet(input_size=11, hidden_size=HIDDEN_SIZE, output_size=3)
        self.trainer = QTrainer(self.model, lr=lr, gamma=GAMMA)
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


def run_experiment(lr):
    agent = Agent(lr)
    game = SnakeGame()
    tracker = ExperimentTracker()
    tracker.start(
        {
            "description": f"Test LR={lr}",
            "lr": lr,
            "gamma": GAMMA,
            "hidden_size": HIDDEN_SIZE,
            "batch_size": BATCH_SIZE,
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
            agent.train_long_memory()
            print(f"LR {lr} | Juego {agent.n_games} - Score: {score}")
            if tracker.log_score(agent.n_games, score):
                tracker.save_model(agent.model)

    tracker.plot_scores()
    tracker.summarize_run()
    pygame.quit()


if __name__ == "__main__":
    for lr in learning_rates:
        run_experiment(lr)
