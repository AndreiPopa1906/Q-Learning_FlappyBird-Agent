# 🐦 Flappy Bird AI: Q-Learning with Deep Reinforcement Learning

This repository contains a fully functional implementation of a **Deep Q-Learning Agent** designed to master the game of Flappy Bird using PyTorch and Gymnasium. Experience the evolution of an AI agent as it learns to navigate the pipes and maximize its reward through trial and error!

## 🚀 Features
- **Deep Q-Network (DQN):** Leverages convolutional layers to process game frames and predict optimal actions.
- **Epsilon-Greedy Exploration:** Balances exploration and exploitation with a decaying epsilon strategy.
- **Replay Buffer:** Efficient training with prioritized experience replay.
- **Reward Shaping:** Encourages survival and rewards pipe passes for faster learning.
- **Target Network Updates:** Improves training stability with periodic hard updates.
- **Advanced Augmentations:**
  - Reward for staying alive.
  - Frame skipping for efficiency.

## 📚 Key Components
- **Custom Neural Network:** Built for processing image inputs with convolutional layers and dense layers for Q-value prediction.
- **Gymnasium Integration:** Plays and trains on the Flappy Bird environment.
- **TensorBoard Logging:** Visualize training metrics like loss, Q-values, and rewards.
- **Save & Load Models:** Easily resume training or evaluate saved models.

## 🔧 How to Use
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/flappy-bird-dqn.git
   cd flappy-bird-dqn
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Train the agent**:
   ```bash
   python main.py
   ```
4. **Watch the agent play**:
   Load the trained model and visualize its performance in the environment.

## 📈 Training Pipeline
- **Step 1:** Initialize the agent with a random policy and collect experiences.
- **Step 2:** Train the agent with mini-batches from the replay buffer.
- **Step 3:** Periodically update the target network to stabilize learning.
- **Step 4:** Log training metrics and save the best-performing model.

## 🌟 Highlights
- Explore how Deep Reinforcement Learning can solve complex environments like Flappy Bird.
- Built-in support for **reward shaping** and **action repetition** to encourage efficient learning.
- Modular and easy-to-extend codebase for experimentation and learning.

## 🏆 Why This Project?
This project demonstrates the power of Deep Reinforcement Learning by applying DQN to a visually simple yet challenging task. It's perfect for understanding the fundamentals of RL, computer vision, and neural network optimization.
