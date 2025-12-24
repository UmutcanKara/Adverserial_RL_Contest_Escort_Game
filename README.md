# Capture & Escort — Multi-Agent Reinforcement Learning Project

## Overview
This project implements a **3 vs 3 competitive game** where two teams learn how to **escort a payload to a goal** while **blocking the opposing team**.

Each team is controlled by artificial agents that **learn from experience** using a technique called **Reinforcement Learning (RL)**.  
The agents are trained using **PPO (Proximal Policy Optimization)** and play against each other in a **self-play** setting.

The goal of the project is not just to win the game, but to demonstrate:
- coordinated team behavior
- competition between teams
- learning from rewards instead of hard-coded rules

---

## The Game (in simple terms)

### Teams
- **Team A**: 3 agents  
- **Team B**: 3 agents  

### Objective
- Each team has **one payload**
- The payload moves **only if agents stay close to it**
- First team to push its payload to the goal **wins**

### Actions (Discrete)
Each agent can choose **one action per step**:
- 0 -> Stay Still
- 1 -> Move Left
- 2 -> Move Right
- 3 -> Move Up
- 4 -> Move Down

### Why this is interesting
- Agents must **cooperate** with teammates
- Agents must **compete** against opponents
- No scripted behavior — everything is learned

---

## How Learning Works (High Level)

### Reinforcement Learning (RL)
Instead of telling agents what to do, we:
1. Let them act in the environment
2. Give them **rewards** for good behavior
3. Penalize bad behavior
4. Let them improve over time

### PPO (Proximal Policy Optimization)
PPO is a stable and widely used learning algorithm that:
- Tries new actions
- Keeps changes small to avoid breaking learned behavior
- Learns from batches of experience

You **do not need to fully understand PPO math** to run or modify this project.

---

## Multi-Agent Setup
- Each team uses **one shared neural network**
- The same model controls all 3 agents on a team
- Agents receive **individual observations** but learn together

This naturally leads to:
- escort roles
- blocking roles
- emergent teamwork

---

## Project Structure
```
.
├── capture_escort.py   # Game Environment
├── PPO.py              # PPO implementation (discrete actions)
├── main.py             # Training + evaluation script
├── README.md           # This file
```



## Running the Project

### 1. Create Virtual Environment

```bash
python3 -m venv ./.venv
source .venv/bin/activate
```

### 2. Install Required Libraries

```bash
pip install -r requirements.txt
```

### 3. Train Agents
```bash
python3 main.py
```

## This Will

- Train both teams using PPO  
- Run evaluation games  
- Print performance statistics  

---

## Output & Evaluation

During evaluation, the program logs:

- Payload progress for both teams  
- Mean reward per team  
- Game winner per episode  

### Example Output

Episode 0 → Winner: A
Final payload progress A: 1.00
Final payload progress B: 0.62

This allows you to:

- Verify agents are actually pushing the payload  
- Compare team performance  
- Plot learning curves if desired  

---

## Why Discrete Actions?

Discrete actions:

- Are easier to debug  
- Reduce training instability  
- Make behavior easier to interpret  

This project originally used continuous actions but was intentionally converted to discrete actions for clarity and stability.

---

## Common Questions

### ❓ What are known pitfalls
- As this project is made to be working alongside with EA algorithms, more often than not degenerate behaviour can be observed in earlier loops of learning
- PPO algorithm and reward functions need optimization.

### ❓ Why are results sometimes noisy?

Because:
- Both teams learn at the same time  
- The environment changes as agents improve  

We reduce this by:
- Lower exploration over time  
- Using deterministic evaluation  
- Smaller learning rates  

---

### ❓ Do I need to understand all the math?

No.

You only need to understand:
- Agents try actions  
- Rewards guide learning  
- Better rewards → better behavior  

The code handles the rest.

---

### ❓ Can I extend this project?

Yes. Easy extensions include:
- Adding an attack action  
- Changing reward weights  
- Increasing team size (e.g., 4v4)  
- Visualizing the game with Pygame  
- Replacing PPO with another algorithm  

---

## Educational Purpose

This project is designed for:
- Students learning reinforcement learning  
- Multi-agent system experiments  
- Demonstrating cooperation vs competition  
- Academic projects and term assignments  

---

## Summary

- 3v3 competitive environment  
- Discrete action PPO  
- Learned teamwork  
- No hard-coded strategies  
- Fully reproducible  

If the agents can **learn to push the payload**, the system is working correctly.