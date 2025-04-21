from collections import deque
import random
import numpy as np


class ReplayMemory:
    def __init__(self, capacity=100_000):
        self.memory = deque(maxlen=capacity)

    def push(self, experience):
        self.memory.append(experience)

    def sample(self, batch_size):
        sample = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*sample)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )

    def __len__(self):
        return len(self.memory)
