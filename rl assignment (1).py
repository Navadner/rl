#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math
import matplotlib.pyplot as plt


class UCB:
    def __init__(self, n_actions):  
        self.n_actions = n_actions
        self.action_counts = np.zeros(n_actions)  
        self.action_rewards = np.zeros(n_actions)  

    def select_action(self, step):
       
        if step < self.n_actions:
            return step  
        
        ucb_values = [
            (self.action_rewards[i] / (self.action_counts[i] + 1e-5)) +  
            math.sqrt(2 * math.log(step + 1) / (self.action_counts[i] + 1e-5))
            for i in range(self.n_actions)
        ]
        return np.argmax(ucb_values)

    def update(self, action, reward):
        
        self.action_counts[action] += 1
        self.action_rewards[action] += reward


def simulate_game(n_steps, n_actions, true_reward_probs):
    ucb = UCB(n_actions)
    total_reward = 0
    rewards = []

    for step in range(n_steps):
        action = ucb.select_action(step)
        reward = 1 if np.random.rand() < true_reward_probs[action] else 0
        ucb.update(action, reward)
        total_reward += reward
        rewards.append(total_reward)

    return rewards, ucb.action_counts


n_steps = 1000  
n_actions = 5  
true_reward_probs = [0.1, 0.3, 0.5, 0.7, 0.9]  

# Run the simulation
rewards, action_counts = simulate_game(n_steps, n_actions, true_reward_probs)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(rewards, label="Cumulative Reward")
plt.xlabel("Steps")
plt.ylabel("Cumulative Reward")
plt.title("UCB Optimization in Game Simulation")
plt.legend()
plt.grid(True)
plt.show()

# Print action counts
print("Action counts:", action_counts)
print("True reward probabilities:", true_reward_probs)


# In[2]:


import numpy as np
import math
import matplotlib.pyplot as plt


class UCB:
    def __init__(self, n_modes):  
        self.n_modes = n_modes
        self.mode_counts = np.zeros(n_modes)  
        self.mode_rewards = np.zeros(n_modes) 

    def select_mode(self, step):
        
        if step < self.n_modes:
            return step  

        ucb_values = [
            (self.mode_rewards[i] / (self.mode_counts[i] + 1e-5)) +
            math.sqrt(2 * math.log(step + 1) / (self.mode_counts[i] + 1e-5))
            for i in range(self.n_modes)
        ]
        return np.argmax(ucb_values)

    def update(self, mode, reward):
        self.mode_counts[mode] += 1
        self.mode_rewards[mode] += reward

def simulate_smart_home(n_steps, n_modes, true_efficiency):
    ucb = UCB(n_modes)
    total_efficiency = 0
    efficiencies = []

    for step in range(n_steps):
        mode = ucb.select_mode(step)
       
        efficiency = np.random.normal(loc=true_efficiency[mode], scale=0.1)
        ucb.update(mode, efficiency)
        total_efficiency += efficiency
        efficiencies.append(total_efficiency)

    return efficiencies, ucb.mode_counts


n_steps = 1000 
n_modes = 4 
true_efficiency = [0.6, 0.7, 0.8, 0.9] 


efficiencies, mode_counts = simulate_smart_home(n_steps, n_modes, true_efficiency)


plt.figure(figsize=(12, 6))
plt.plot(efficiencies, label="Cumulative Energy Efficiency")
plt.xlabel("Steps")
plt.ylabel("Cumulative Energy Efficiency")
plt.title("UCB Optimization in Smart Home Energy Usage")
plt.legend()
plt.grid(True)
plt.show()


print("Mode counts:", mode_counts)
print("True mode efficiencies:", true_efficiency)


# #  3) Develop a Chess-like game using PAC (Probably Approximately Correct) algorithm where the Problem set-up is as follows:**  
# 
# Problem Setup:  
# Game Environment: 
# - A simplified 4x4 chessboard.  
# - Pieces: King, Queen, and Pawn.  
# - Objective: Move pieces to maximize a reward function (e.g., capturing opponent pieces or reaching specific board positions).  
# 
# PAC Learning:  
# - The AI evaluates a set of possible moves (hypotheses) for each turn.  
# - A hypothesis is "probably approximately correct" if it selects moves that are likely to result in near-optimal outcomes within a specified error margin (ε) and confidence level (δ).  
# 
# Goal:  
# Train the AI to select moves that maximize the expected game reward while maintaining ε (error) and δ (confidence) guarantees.  
# 
# Key Components: 
# 1. ChessBoard Class: 
#    - Represents a simplified 4x4 chessboard.  
#    - Tracks player and opponent pieces.  
# 2. Reward Function: 
#    - Rewards capturing opponent pieces or moving to specific positions.  
# 3. Move Generation: 
#    - Generates valid moves for each piece (King, Queen, Pawn).  
# 4. PAC Algorithm:  
#    - Estimates the expected reward for each move by sampling.  
#    - Selects the move with the highest average reward, satisfying PAC conditions.  
# 5. Simulation:  
#    - Simulates turns and updates the board state.  
# 
# Run Simulation:  
# - Execute the script to observe the AI selecting moves to maximize rewards.  
# - Adjust ε (error) and δ (confidence) to modify the PAC learning guarantees.  
# 
# ---

# In[5]:


import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import random

# Define the board and pieces
class ChessBoard:
    def __init__(self):
        self.board = np.zeros((4, 4), dtype=int)  # 4x4 grid
        self.king_pos = (3, 0)  # King's initial position
        self.pawn_positions = [(0, 1), (1, 3), (2, 2)]  # Pawns' initial positions
        self.place_pieces()

    def place_pieces(self):
        self.board[self.king_pos] = 1  # King represented as 1
        for pos in self.pawn_positions:
            self.board[pos] = -1  # Pawns represented as -1

    def get_features(self):
        # Flatten board as feature vector
        return self.board.flatten()

    def move_king(self, new_pos):
        x, y = self.king_pos
        self.board[x, y] = 0  # Clear old king position
        self.king_pos = new_pos
        x, y = new_pos
        self.board[x, y] = 1  # Set new king position

    def is_valid_move(self, pos):
        x, y = pos
        return 0 <= x < 4 and 0 <= y < 4 and self.board[x, y] != 1  # Inside bounds and not the king's current position

    def generate_king_moves(self):
        x, y = self.king_pos
        moves = [(x + i, y + j) for i in [-1, 0, 1] for j in [-1, 0, 1] if (i, j) != (0, 0)]
        return [move for move in moves if self.is_valid_move(move)]

# Reward function
def reward_function(board, move):
    x, y = move
    if board[x, y] == -1:  # Capture a pawn
        return 10
    else:  # Move to an empty space
        return 1

# Generate training data
def generate_training_data(n_samples):
    X = []
    y = []
    for _ in range(n_samples):
        board = ChessBoard()
        moves = board.generate_king_moves()
        optimal_move = None
        max_reward = -np.inf
        for move in moves:
            reward = reward_function(board.board, move)
            if reward > max_reward:
                max_reward = reward
                optimal_move = move
        X.append(board.get_features())
        y.append(optimal_move)
    return np.array(X), np.array(y)

# Train a PAC model (decision tree classifier)
def train_pac_model(X, y):
    # Flatten move labels for multi-output classification
    y_flat = [x * 4 + y for x, y in y]
    model = DecisionTreeClassifier(max_depth=5)
    model.fit(X, y_flat)
    return model

# Predict a move
def predict_move(model, board):
    move_flat = model.predict([board.get_features()])[0]
    return divmod(move_flat, 4)

# Simulate a game
def simulate_game(model, n_steps=10):
    board = ChessBoard()
    for step in range(n_steps):
        print(f"Step {step + 1}:")
        print(board.board)
        move = predict_move(model, board)
        print(f"King moves to: {move}")
        board.move_king(move)

# Main function
if __name__ == "__main__":
    # Generate training data
    X, y = generate_training_data(500)

    # Train PAC model
    model = train_pac_model(X, y)

    # Evaluate model accuracy
    y_flat = [x * 4 + y for x, y in y]
    predictions = model.predict(X)
    print(f"Model Accuracy: {accuracy_score(y_flat, predictions) * 100:.2f}%")

    # Simulate a game
    simulate_game(model)


# In[ ]:




