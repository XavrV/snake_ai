import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepQNet(nn.Module):
    """
    Red neuronal profunda para Q-Learning con dos capas ocultas.
    """

    def __init__(
        self, input_size: int, hidden_size1: int, hidden_size2: int, output_size: int
    ) -> None:
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.linear3(x)

    def save(self, file_name: str = "model_deep.pth") -> None:
        torch.save(self.state_dict(), file_name)
