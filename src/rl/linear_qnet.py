import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any


class LinearQNet(nn.Module):
    """
    Red neuronal Q-lineal para aprendizaje por refuerzo.

    Esta red consta de dos capas lineales:
      - La primera capa transforma el estado de entrada a una representación oculta,
        seguida de una activación ReLU.
      - La segunda capa produce los valores Q correspondientes a cada acción.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.linear1: nn.Linear = nn.Linear(input_size, hidden_size)
        self.linear2: nn.Linear = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Realiza el pase hacia adelante a través de la red neuronal.

        Args:
            x (torch.Tensor): El tensor de entrada que representa el estado.

        Returns:
            torch.Tensor: Los valores Q para cada acción.
        """
        x = F.relu(self.linear1(x))
        return self.linear2(x)

    def save(self, file_name: str = "model.pth") -> None:
        """
        Guarda el estado actual del modelo en un archivo.

        Args:
            file_name (str): Nombre del archivo para almacenar el modelo. Por defecto es "model.pth".
        """
        torch.save(self.state_dict(), file_name)
