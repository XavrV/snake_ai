# Snake AI

Proyecto de inteligencia artificial para jugar al clásico juego Snake usando aprendizaje por refuerzo.

## Estructura del Proyecto

- **train.py:** Archivo principal para iniciar el entrenamiento.
- **src/**
  - **game/**
    - `snake_game.py`: Lógica del juego.
  - **rl/**
    - `linear_qnet.py`: Modelo de red neuronal.
    - `trainer.py`: Lógica de entrenamiento.
    - `memory.py`: Manejo de la memoria de experiencias.
  - **state/**
    - `state_representation.py`: Obtención y procesamiento del estado del juego.

## Requisitos

- Python 3.10+
- [PyTorch](https://pytorch.org/)
- Pygame
- NumPy

Instalación:
```sh
pip install -r requirements.txt
```

## Uso

Para iniciar el entrenamiento:
```sh
python train.py
```

## Notas
