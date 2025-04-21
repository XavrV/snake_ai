# src/utils/plotter.py
import matplotlib.pyplot as plt
from IPython.display import clear_output

scores = []
means = []


def plot(scores_list):
    scores.append(scores_list[-1])
    means.append(sum(scores[-100:]) / len(scores[-100:]))

    clear_output(wait=True)
    plt.figure(figsize=(10, 5))
    plt.title("Progreso del Entrenamiento")
    plt.xlabel("Número de Juegos")
    plt.ylabel("Score")
    plt.plot(scores, label="Score")
    plt.plot(means, label="Promedio (últimos 100)")
    plt.legend()
    plt.grid(True)
    plt.show()
