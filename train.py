import torch
import pygame
from src.game.snake_game import SnakeGame
from src.state.state_representation import get_state
from src.rl.linear_qnet import LinearQNet
from src.rl.trainer import QTrainer
from src.rl.memory import ReplayMemory
import random

# Parámetros globales
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:
    """
    Agente que utiliza aprendizaje por refuerzo.
    """

    def __init__(self) -> None:
        self.model = LinearQNet(input_size=11, hidden_size=256, output_size=3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=0.9)
        self.memory = ReplayMemory(MAX_MEMORY)
        self.epsilon = 0  # Control de exploración
        self.n_games = 0

    def get_action(self, state):
        """
        Selecciona una acción basada en una política de exploración y explotación.
        """
        self.epsilon = 80 - self.n_games  # Ejemplo de decaimiento de epsilon
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
        """
        Entrena la red con un solo paso.
        """
        self.trainer.train_step(state, action, reward, next_state, done)

    def remember(self, state, action, reward, next_state, done):
        """
        Guarda la experiencia en la memoria.
        """
        self.memory.push((state, action, reward, next_state, done))

    def train_long_memory(self):
        """
        Entrena la red utilizando un batch de experiencias.
        """
        if len(self.memory) > BATCH_SIZE:
            batch = self.memory.sample(BATCH_SIZE)
        else:
            batch = self.memory.sample(len(self.memory))
        self.trainer.train_step(*batch)


def train():
    """
    Función principal de entrenamiento.
    """
    agent = Agent()
    game = SnakeGame()

    while True:
        state_old = get_state(game)
        action = agent.get_action(state_old)
        reward, done, score = game.play_step(action)
        game.render_pygame(speed=20)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        state_new = get_state(game)
        agent.train_short_memory(state_old, action, reward, state_new, done)
        agent.remember(state_old, action, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            print(f"Juego {agent.n_games} - Score: {score}")


if __name__ == "__main__":
    train()
