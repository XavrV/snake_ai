import random
from collections import deque
from enum import Enum
from dataclasses import dataclass
import pygame

# Colores
WHITE = (255, 255, 255)
RED = (200, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20  # Tamaño de cada celda del tablero


# Direcciones posibles
class Direction(Enum):
    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3


# Punto en el tablero
@dataclass
class Point:
    x: int
    y: int


# Juego Snake
class SnakeGame:
    def __init__(self, width=10, height=10):
        self.width = width
        self.height = height

        pygame.init()
        self.display = pygame.display.set_mode(
            (self.width * BLOCK_SIZE, self.height * BLOCK_SIZE)
        )
        pygame.display.set_caption("Snake IA")
        self.clock = pygame.time.Clock()

        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.width // 2, self.height // 2)
        self.snake = deque(
            [
                self.head,
                Point(self.head.x - 1, self.head.y),
                Point(self.head.x - 2, self.head.y),
            ]
        )
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        while True:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            self.food = Point(x, y)
            if self.food not in self.snake:
                break

    def play_step(self, action):
        self.frame_iteration += 1
        self._move(action)
        self.snake.appendleft(self.head)

        # Check collision
        if self._is_collision() or self.frame_iteration > 20 * len(self.snake):
            return -10, True, self.score

        # Check if food eaten
        if self.head == self.food:
            self.score += 1
            self._place_food()
            reward = 10
        else:
            self.snake.pop()
            reward = 0

        return reward, False, self.score

    def _move(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if action == [1, 0, 0]:  # Straight
            new_dir = clock_wise[idx]
        elif action == [0, 1, 0]:  # Right turn
            new_dir = clock_wise[(idx + 1) % 4]
        else:  # Left turn
            new_dir = clock_wise[(idx - 1) % 4]

        self.direction = new_dir

        x, y = self.head.x, self.head.y
        if self.direction == Direction.RIGHT:
            x += 1
        elif self.direction == Direction.LEFT:
            x -= 1
        elif self.direction == Direction.UP:
            y -= 1
        elif self.direction == Direction.DOWN:
            y += 1

        self.head = Point(x, y)

    def _is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt.x < 0 or pt.x >= self.width or pt.y < 0 or pt.y >= self.height:
            return True
        if pt in list(self.snake)[1:]:
            return True
        return False

    def render_pygame(self, speed=10):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(
                self.display,
                GREEN,
                pygame.Rect(
                    pt.x * BLOCK_SIZE, pt.y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE
                ),
            )

        pygame.draw.rect(
            self.display,
            RED,
            pygame.Rect(
                self.head.x * BLOCK_SIZE,
                self.head.y * BLOCK_SIZE,
                BLOCK_SIZE,
                BLOCK_SIZE,
            ),
        )
        pygame.draw.rect(
            self.display,
            WHITE,
            pygame.Rect(
                self.food.x * BLOCK_SIZE,
                self.food.y * BLOCK_SIZE,
                BLOCK_SIZE,
                BLOCK_SIZE,
            ),
        )

        pygame.display.flip()
        self.clock.tick(speed)
