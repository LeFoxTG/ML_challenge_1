import torch
import torch.nn as nn

class AtariActorCritic(nn.Module):
    def __init__(self, n_actions):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.fc = nn.Linear(64 * 7 * 7, 512)

        self.actor = nn.Linear(512, n_actions)
        self.critic = nn.Linear(512, 1)

    def forward(self, x):
        x = x / 255.0 if x.max() > 1 else x
        feats = self.cnn(x)
        feats = torch.relu(self.fc(feats))

        return self.actor(feats), self.critic(feats).squeeze(-1)