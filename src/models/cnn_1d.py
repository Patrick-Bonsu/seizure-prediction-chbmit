import torch
import torch.nn as nn


class CNN1D(nn.Module):
    """
    Simple 1D CNN baseline for windowed EEG.
    Input:  (B, C, L)  e.g., (B, 23, 1280)
    Output: logits (B, 2)  for CrossEntropyLoss
    """

    def __init__(self, in_channels: int = 23, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # L/2

            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # L/4

            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            # Makes the model robust to different L by pooling to a fixed length
            nn.AdaptiveAvgPool1d(output_size=16),  # (B, 128, 16)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),                  # (B, 128*16)
            nn.Dropout(dropout),
            nn.Linear(128 * 16, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),   # logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x
