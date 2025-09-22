# A2C-GAE-PPO

This repository contains an implementation of **Advantage Actor-Critic (A2C)** with **Generalized Advantage Estimation (GAE)** and an extension to **Proximal Policy Optimization (PPO)**.  

The code is designed to work with both classic control tasks and **Atari environments** (with convolutional feature extraction).  

---

## Features
- A2C with GAE for variance reduction  
- PPO extension with clipping and minibatch optimization  
- Support for **vectorized environments** (`SyncVectorEnv`)  
- Atari mode with convolutional feature extractor  
- Training and evaluation loop with logging  

---

## Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/augustoafleal/a2c-gae-ppo.git
cd A2C-GAE-PPO
pip install -r requirements.txt
