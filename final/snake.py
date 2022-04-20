import torch
import random
import numpy as np
from collections import deque
import import_ipynb
from game import SnakeGameAI, Direction, Point
from model import DQN, DQN_trainer
from helper import plot
import os
import sys

Batch_size = 1024
lr = 1e-3

class Agent:
    def __init__(self,model):
        self.ngames = 0 
        self.initial_epsilon = 0.4 # randomness
        self.epsilon_decrement = 0.005
        self.gamma = 0.8 # discount rate
        self.mem = deque() 
        self.model = model
        self.trainer = DQN_trainer(self.model, lr = lr, gamma = self.gamma)
    
    def get_state(self,game):
        head = game.snake[0] # get the position of the head
        
        # direction and angle clockwise
        cw_dir = [game.direction == Direction.RIGHT,
                  game.direction == Direction.DOWN,
                  game.direction == Direction.LEFT,
                  game.direction == Direction.UP
        ]
        cw_ang =  np.array([0, np.pi/2, np.pi, -np.pi/2])
        
        get_point = lambda i : Point(
            head.x + 20*np.cos(cw_ang[(cw_dir.index(True)+ i)% 4]),
            head.y + 20*np.sin(cw_ang[(cw_dir.index(True)+ i)% 4]))
        state = [
            # check if snake is hitting boundary
            game.is_collision(get_point(0)), #front
            game.is_collision(get_point(-1)),  #left
            game.is_collision(get_point(1)),   #right

            
            # move direction
            cw_dir[3],
            cw_dir[1],
            cw_dir[2],
            cw_dir[0],
            
            # food location
            head.y > game.food.y, # food up
            head.y < game.food.y, # food down
            head.x > game.food.x, # food left
            head.x < game.food.x # food right   
            ]
        
        return np.array(state, dtype = int)
    
    def remember(self,state,move,reward,next_state,gg):
        self.mem.append((state,move,reward,next_state,gg))
    
    def train_short_mem(self, state, move, reward, next_state, gg):   # 1 step
        self.trainer.train_step(state, move, reward, next_state, gg)
    
    def train_long_mem(self): # long term
        if len(self.mem) < Batch_size:
            sample = self.mem
        else:
            sample = random.sample(self.mem, Batch_size)
        
        for state, move, reward, next_state, gg in sample:
            self.trainer.train_step(state, move, reward, next_state, gg)

    
    def get_move(self,state):
        epsilon = self.initial_epsilon - self.ngames * self.epsilon_decrement # randomness
        curr_move = [0,0,0]
        if random.random() < epsilon:
            i = random.randint(0,2)
            curr_move[i] = 1
        else:
            torch_state = torch.tensor(state, dtype = torch.float)
            pred = self.model(torch_state)
            i = torch.argmax(pred).item()
            curr_move[i] = 1
        return curr_move 
    
    def test_play(self,state): 
        self.model.eval()
        curr_move = [0,0,0]
        torch_state = torch.tensor(state, dtype = torch.float)
        pred = self.model(torch_state)
        i = torch.argmax(pred).item()
        curr_move[i] = 1
        
        return curr_move 
    
def train():
    scores = []
    total_score = 0
    mean_scores = []
    record = 0
    train_model = DQN(11,256,3)
    #train_model = DQN(11,512,3)
    #train_model = DQN(11,256,128,3)
    #train_model = DQN(11,256,64,32,3)
    agent = Agent(train_model)
    game = SnakeGameAI()
    
    
    while True:
        curr_state = agent.get_state(game) # get the current state of the game
        curr_move = agent.get_move(curr_state) # get the move for the current state
        reward, gg, score = game.play_step(curr_move) # get the reward, status of the game and score from the game
        new_state = agent.get_state(game) # get the new state of the game
        
        agent.train_short_mem(curr_state, curr_move, reward, new_state, gg) # train from 1 step
        
        agent.remember(curr_state, curr_move, reward, new_state, gg) # store all in the deque
        
        if gg: # if game over, perform experience replay to train on all previous experiences
            game.reset()
            agent.ngames += 1 
            agent.train_long_mem() 
            
            if record < score:
                record = score
                torch.save(agent.model, "trained_model/current_model_" + "record" + ".pth")
            if agent.ngames % 50 == 0:
                torch.save(agent.model, "trained_model/current_model_" + str(agent.ngames) + ".pth")
            
            scores.append(score)
            total_score += score
            mean_score = total_score / agent.ngames
            mean_scores.append(mean_score)
            plot(scores,mean_scores)

def test():
    
    model = torch.load('trained_model/current_model_record.pth',)
    agent = Agent(model)
    game = SnakeGameAI()
    
    
    while True:
        curr_state = agent.get_state(game)
        curr_move = agent.test_play(curr_state)
        reward, gg, score = game.play_step(curr_move)
        
        if gg:
            game.reset()

def main(mode):

    if mode == 'test':
        test()

    elif mode == 'train':
        if not os.path.exists('trained_model/'):
            os.mkdir('trained_model/')
        train()


if __name__ == "__main__":
    main(sys.argv[1])