import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class DQN(nn.Module):
    def __init__(self,input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = self.fc2(X)

        return X

class DQN_trainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr = self.lr)
        self.criterion = nn.MSELoss()
        

    def train_step(self, state, move, reward, new_state, gg):
        state = torch.tensor(state, dtype = torch.float)
        move = torch.tensor(move, dtype = torch.float)
        reward = torch.tensor(reward, dtype = torch.float)
        new_state = torch.tensor(new_state, dtype = torch.float)
        
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            move = torch.unsqueeze(move, 0)
            reward = torch.unsqueeze(reward, 0)
            new_state = torch.unsqueeze(new_state, 0)
            gg = (gg, )
        
        pred = self.model(state) # predict current Q value with the current state
        
        target = pred.clone()
        
        for i in range(len(gg)):
            if gg[i]: # if game over
                Q_new = reward[i]
            else:
                Q_new = reward[i] + self.gamma * torch.max(self.model(new_state[i]))
                
            target[i][torch.argmax(move).item()] = Q_new
            
        self.optimizer.zero_grad()
        
        loss = self.criterion(pred, target)
        loss.backward()
        
        self.optimizer.step()



