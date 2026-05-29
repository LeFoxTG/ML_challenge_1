import torch
import torch.nn as nn

class GAILDiscriminator ( nn.Module ) :
    """
    Takes a stacked - frame observation ( and optionally a one - hot encoded
    action ) and outputs P ( expert | s, a ) in (0, 1) .
    Using observation only ( obs - only variant ) is simpler and often
    sufficient for image - based environments .
    """
    def __init__ ( self, n_actions : int, use_action : bool = False ) :
        super().__init__()
        self.use_action = use_action
        # Shared CNN - same architecture as the policy backbone
        self.cnn = nn.Sequential (
            nn.Conv2d (4, 32, kernel_size =8, stride =4), nn.ReLU(),
            nn.Conv2d (32, 64, kernel_size =4, stride =2), nn.ReLU(),
            nn.Conv2d (64, 64, kernel_size =3, stride =1), nn.ReLU(),
            nn.Flatten(),
        )
        cnn_out = 64 * 7 * 7 # 3136
        fc_in = cnn_out + n_actions if use_action else cnn_out

        self.fc = nn.Sequential (
            nn.Linear ( fc_in, 512), nn.Tanh(),
            nn.Linear (512, 1), nn.Sigmoid(),
        )

    def forward ( self, obs, actions_onehot = None ) :
        feats = self.cnn ( obs )
        if self.use_action and actions_onehot is not None :
            feats = torch.cat ([ feats, actions_onehot ], dim = -1)
        return self.fc ( feats ).squeeze ( -1)