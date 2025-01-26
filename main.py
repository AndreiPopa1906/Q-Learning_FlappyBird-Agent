import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import gymnasium as gym
import flappy_bird_gymnasium
import torch
import glob
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import cv2
import os
import time
import warnings
from torch.utils.tensorboard import SummaryWriter

# Suppress specific gymnasium warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium.utils.passive_env_checker")

# -------------------
# Hyperparameters
# -------------------
ACTIONS = 2  # 0: Do nothing, 1: Flap
GAMMA = 0.99

# Epsilon settings (for epsilon-greedy)
INITIAL_EPSILON = 1.0   # Start with full exploration
FINAL_EPSILON   = 0.01  # Final value after decay
EPSILON_DECAY   = 50000 # Decay over 50k steps

# Replay buffer and training
REPLAY_MEMORY  = 50000  # Increased replay memory
BATCH_SIZE     = 32     # Increased batch size for more stable training
LEARNING_RATE  = 1e-4
TARGET_UPDATE_FREQ = 1000    # Update target network every 1000 steps
MAX_EPISODES      = 10000    # You can increase this if you have time

# Frame skipping
ACTION_REPEAT = 2  # Number of frames to repeat an action

# Warmup: collect experiences without training
LEARNING_STARTS = 5000

# Reward shaping
REWARD_PER_FRAME_ALIVE = 0.1
REWARD_FOR_PASSING_PIPE = 5.0  # You can tune this value

# -------------------
# Neural Network
# -------------------
class FlappyBirdNetwork(nn.Module):
    def __init__(self):
        super(FlappyBirdNetwork, self).__init__()
        # Convolutions over the images
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=2)  # [B, 32, 20, 20]
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1) # [B, 64, 10, 10]
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1) # [B, 64, 10, 10]
        self.bn3   = nn.BatchNorm2d(64)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 10 * 10, 512)
        self.fc2 = nn.Linear(512, ACTIONS)

    def forward(self, x):
        # x shape: [batch, 4, 80, 80]
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        return self.fc2(x)  # Q-values for each action

# -------------------
# Agent
# -------------------
class FlappyBirdAgent:
    def __init__(self, load_model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Main DQN and Target DQN
        self.model = FlappyBirdNetwork().to(self.device)
        self.target_model = FlappyBirdNetwork().to(self.device)
        # If we have a path to load from:
        if load_model_path is not None and os.path.exists(load_model_path):
            print(f"Loading model weights from {load_model_path}...")
            self.model.load_state_dict(torch.load(load_model_path, map_location=self.device))

        self.update_target_network()

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100000, gamma=0.1)

        # Loss function
        self.criterion = nn.MSELoss()

        # Replay buffer
        self.memory = deque(maxlen=REPLAY_MEMORY)

        # Epsilon and counters
        self.epsilon       = INITIAL_EPSILON
        self.step_count    = 0  # Counts every environment step
        self.episode_count = 0
        self.total_steps   = 0  # Total steps across episodes

        # Track the best reward so far
        self.best_reward   = float('-inf')

        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir='D:/FlappyBirdOutput')

        # Create directory for saving models if it doesn't exist
        if not os.path.exists('saved_networks'):
            os.makedirs('saved_networks')

    def update_target_network(self):
        """Hard update: copy weights from main model to target model."""
        self.target_model.load_state_dict(self.model.state_dict())

    def preprocess_frame(self, frame):
        # RGB -> Gray
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Resize to 80x80
        frame = cv2.resize(frame, (80, 80))
        frame = frame.astype(np.float32) / 255.0

        # Add channel dimension [1, 80, 80]
        return np.expand_dims(frame, axis=0)

    def select_action(self, state):
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randrange(ACTIONS)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).to(self.device)  # [4, 80, 80]
                state_tensor = state_tensor.unsqueeze(0)                 # [1, 4, 80, 80]
                q_values = self.model(state_tensor)
                return torch.argmax(q_values).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self):
        """Sample from replay memory and update the network."""
        if len(self.memory) < BATCH_SIZE or self.step_count < LEARNING_STARTS:
            return None, None

        minibatch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states     = torch.FloatTensor(np.array(states)).to(self.device)       # [B, 4, 80, 80]
        actions    = torch.LongTensor(actions).unsqueeze(1).to(self.device)    # [B, 1]
        rewards    = torch.FloatTensor(rewards).to(self.device)                # [B]
        next_states= torch.FloatTensor(np.array(next_states)).to(self.device)  # [B, 4, 80, 80]
        dones      = torch.FloatTensor(dones).to(self.device)                  # [B]

        # Current Q-values
        current_q = self.model(states).gather(1, actions).squeeze()  # [B]

        # Double DQN target
        with torch.no_grad():
            # Choose best action using main network
            next_actions = self.model(next_states).argmax(dim=1, keepdim=True)       # [B, 1]
            # Evaluate that action using target network
            next_q = self.target_model(next_states).gather(1, next_actions).squeeze()# [B]

        # Q target
        target_q = rewards + GAMMA * next_q * (1 - dones)

        # Loss
        loss = self.criterion(current_q, target_q.detach())

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # (Optional) step the LR scheduler
        self.scheduler.step()

        # Log to TensorBoard every 1000 steps
        if self.step_count % 1000 == 0:
            self.writer.add_scalar('Loss', loss.item(), self.step_count)
            self.writer.add_scalar('Avg Target Q', target_q.mean().item(), self.step_count)
            self.writer.add_scalar('Epsilon', self.epsilon, self.step_count)

        return loss.item(), target_q.mean().item()

    def decay_epsilon(self):
        """Linear decay of epsilon until it reaches FINAL_EPSILON."""
        if self.epsilon > FINAL_EPSILON:
            old_epsilon = self.epsilon
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EPSILON_DECAY
            self.epsilon = max(self.epsilon, FINAL_EPSILON)
            print(f"Decaying Epsilon from {old_epsilon:.4f} to {self.epsilon:.4f} at step {self.step_count}")

    def save_model(self, model_path="saved_networks/flappy_bird_dqn_best.pth"):
        """Save the model to a fixed file."""
        torch.save(self.model.state_dict(), model_path)
        print(f"Best model saved to {model_path}.")

    def play_game(self, env, render=False):
        """Run one episode. Returns total_reward, episode_steps."""
        obs, _ = env.reset()
        obs = self.preprocess_frame(obs)  # shape: [1, 80, 80]

        # Initialize state by stacking the same frame 4 times
        state = np.concatenate([obs for _ in range(4)], axis=0)  # [4, 80, 80]

        total_reward = 0
        episode_steps = 0

        while True:
            if render:
                env.render()

            action = self.select_action(state)

            accumulated_reward = 0
            done = False

            for _ in range(ACTION_REPEAT):
                next_obs, reward, terminated, truncated, info = env.step(action)

                # ------------------
                # Reward Shaping
                # ------------------
                reward += REWARD_PER_FRAME_ALIVE

                if "pipe_passed" in info and info["pipe_passed"]:
                    reward += REWARD_FOR_PASSING_PIPE
                    print(f"[Step {self.step_count}] pipe_passed => +{REWARD_FOR_PASSING_PIPE} bonus")

                accumulated_reward += reward

                if terminated or truncated:
                    done = True
                    break

            next_obs = self.preprocess_frame(next_obs)  # shape: [1, 80, 80]

            next_state = np.concatenate((state[1:], next_obs), axis=0)  # [4, 80, 80]

            self.store_transition(state, action, accumulated_reward, next_state, float(done))

            state = next_state

            total_reward += accumulated_reward
            episode_steps += 1
            self.step_count += 1
            self.total_steps += 1

            loss, avg_target_q = self.train_step()

            self.decay_epsilon()

            if self.step_count % TARGET_UPDATE_FREQ == 0:
                self.update_target_network()
                print(f"[Step {self.step_count}] Target network updated.")

            if loss is not None and self.step_count % 100 == 0:
                print(f"[Step {self.step_count}] Loss: {loss:.4f}, AvgQ: {avg_target_q:.4f}, Eps: {self.epsilon:.4f}")

            if done:
                break

        self.episode_count += 1
        self.writer.add_scalar('Episode Reward', total_reward, self.episode_count)

        return total_reward, episode_steps

# -------------------
# Main Training Loop
# -------------------
if __name__ == "__main__":
    best_model_path = "saved_networks/flappy_bird_dqn_step_120000.pth"

    env = gym.make("FlappyBird-v0", render_mode="human")

    agent = FlappyBirdAgent(load_model_path=best_model_path)

    print("Starting training from the pretrained model...")
    start_time = time.time()

    for episode in range(1, MAX_EPISODES + 1):
        total_reward, episode_steps = agent.play_game(env, render=False)
        print(f"Episode {episode} - Reward: {total_reward:.2f}, Steps: {episode_steps}, Epsilon: {agent.epsilon:.4f}")

        if total_reward > agent.best_reward:
            agent.best_reward = total_reward
            agent.save_model("saved_networks/flappy_bird_dqn_best.pth")

    env.close()
    agent.writer.close()
    end_time = time.time()
    print(f"Training completed in {(end_time - start_time) / 60:.2f} minutes.")

