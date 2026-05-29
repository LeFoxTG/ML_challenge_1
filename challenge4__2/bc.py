import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils import data
from torch.utils.data import DataLoader , TensorDataset

from challenge3__2.env import make_env
from challenge3__2.model import AtariActorCritic

def train_bc ( env_id : str , demos_path : str = "demos.npz" ,
    n_epochs : int = 20 , batch_size : int = 256 ,
    lr : float = 1e-4 , device : str = "cpu", output_path : str = "bc_policy.pt" ) :
    """
    Supervised imitation : minimise cross - entropy between
    demonstrations and policy logits .
    """
    data = np.load ( demos_path )
    obs_t = torch.tensor ( data [ "observations" ] , dtype = torch.float32 )
    act_t = torch.tensor ( data [ "actions" ] , dtype = torch.long )
    dataset = TensorDataset ( obs_t , act_t )
    loader = DataLoader ( dataset , batch_size = batch_size , shuffle = True )
    env = make_env(env_id)
    n_actions = env.action_space.n
    env.close()

    model = AtariActorCritic ( n_actions ).to(device)
    optimizer = optim.Adam ( model.parameters(), lr = lr )
    criterion = nn.CrossEntropyLoss ()
    
    best_loss = float("inf")

    for epoch in range ( n_epochs ) :
        total_loss = 0.0
        for obs_b , act_b in loader :
            obs_b , act_b = obs_b.to(device), act_b.to(device)
            logits , _ = model(obs_b)
            loss = criterion(logits, act_b )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / len ( loader )
        print (f"BC epoch { epoch +1}/{n_epochs} loss ={avg:.4f}")
        if avg <= best_loss:
            best_loss = avg
            torch.save(
                model.state_dict(),
                output_path
            )
            print(
                f"New best BC model "
                f"(loss={avg:.4f})"
            )
    
    torch.save ( model.state_dict() , output_path )
    return model