# Debugging Environments for RL Algorithms

## Introduction

Reinforcement Learning (RL) algorithms are often tested on standard environments like those from OpenAI Gymnasium's classic control suite. However, these environments can be noisy and may not help in pinpointing specific issues within your RL implementation. To address this, we've created a set of minimalistic environments designed to isolate and test specific components of an RL agent, such as the value network, policy network, backpropagation, and reward discounting mechanisms.

By progressively adding complexity, these custom environments help you isolate and identify specific issues within your RL algorithms. They serve as valuable tools for debugging and verifying the correctness of various components, ensuring that your RL agent functions as intended before deploying it to more complex tasks.

These environments are extremely lightweight and are intended to help you quickly identify and fix bugs in your RL algorithms. When implemented correctly, agents should learn optimal policies in these environments within seconds.

## Environments Overview

### 1. **OneActionZeroObsEnv**

- **Observation Space:** Always zero.
- **Action Space:** Only one action.
- **Episode Length:** One timestep.
- **Reward:** Constant +1 every timestep.

**Purpose:**  
Isolates the **value network**. The agent should learn that the value of the only observation it ever sees is 1. If the agent fails here, it indicates issues with the value loss calculation or the optimizer.

---

### 2. **OneActionRandomObsEnv**

- **Observation Space:** Randomly +1 or -1.
- **Action Space:** Only one action.
- **Episode Length:** One timestep.
- **Reward:** Observation-dependent (+1 or -1).

**Purpose:**  
Tests **backpropagation through the network**. The agent should learn to predict the value of the state based on the observation. Failure indicates problems with backpropagation or network connectivity.

---

### 3. **OneActionTwoObsEnv**

- **Observation Space:** 0 at the first timestep, 1 at the second.
- **Action Space:** Only one action.
- **Episode Length:** Two timesteps.
- **Reward:** +1 at the end of the episode.

**Purpose:**  
Examines **reward discounting**. The agent should learn the value of the initial observation by considering future rewards. Failure here suggests issues with reward discounting or temporal aspects of the value function.

---

### 4. **TwoActionsZeroObsEnv**

- **Observation Space:** Always zero.
- **Action Space:** Two actions.
- **Episode Length:** One timestep.
- **Reward:** Action-dependent (+1 for action 0, -1 for action 1).

**Purpose:**  
Tests the **policy network**. The agent should learn to select the better action based on the reward. If the agent can't learn this, there may be problems with the advantage calculations, policy loss, or policy updates.

---

### 5. **TwoActionsRandomObsEnv**

- **Observation Space:** Randomly +1 or -1.
- **Action Space:** Two actions.
- **Episode Length:** One timestep.
- **Reward:** Dependent on both action and observation (+1 if action matches observation, -1 otherwise).

**Purpose:**  
Assesses the **interaction between policy and value networks**. The agent should learn to choose actions based on the observation. This environment can reveal issues in experience batching or network synchronization.

## Usage Instructions

### Setup

1. **Install Dependencies:**

   Ensure you have Gymnasium installed:

   ```bash
   pip install gymnasium
   ```

2. **Add the Environments to Your Project:**

   Save each environment class in a Python file within your project directory, for example, `custom_envs.py`.

3. **Register the Environments:**

   In your main script or an initialization file, register the environments with Gymnasium:

   ```python
   from gymnasium.envs.registration import register

   register(
       id='OneActionZeroObsEnv-v0',
       entry_point='custom_envs:OneActionZeroObsEnv',
   )

   register(
       id='OneActionRandomObsEnv-v0',
       entry_point='custom_envs:OneActionRandomObsEnv',
   )

   register(
       id='OneActionTwoObsEnv-v0',
       entry_point='custom_envs:OneActionTwoObsEnv',
   )

   register(
       id='TwoActionsZeroObsEnv-v0',
       entry_point='custom_envs:TwoActionsZeroObsEnv',
   )

   register(
       id='TwoActionsRandomObsEnv-v0',
       entry_point='custom_envs:TwoActionsRandomObsEnv',
   )
   ```

   Replace `'custom_envs'` with the actual module name where the environment classes are defined.

### Running an Environment

Here's how you can instantiate and run an environment:

```python
import gymnasium as gym

# Example with OneActionZeroObsEnv
env = gym.make('OneActionZeroObsEnv-v0')
observation, info = env.reset()

done = False
while not done:
    action = env.action_space.sample()  # Since there's only one action
    observation, reward, done, truncated, info = env.step(action)
    # Your agent's learning code here

env.close()
```

### Integrating with Your RL Agent

Replace the random action selection with your agent's action decision. Use these environments to test specific components of your agent:

- **Value Network Testing:** Use `OneActionZeroObsEnv` and `OneActionRandomObsEnv` to ensure your value network learns the correct state values.
- **Policy Network Testing:** Use `TwoActionsZeroObsEnv` to verify that your policy network selects actions that maximize rewards.
- **Policy-Value Interaction Testing:** Use `TwoActionsRandomObsEnv` to test the agent's ability to learn optimal policies in state-dependent scenarios.

## Acknowledgements

These environments are inspired by ideas presented in [Andy Jones's amazing blog post on RL debugging](https://andyljones.com/posts/rl-debugging.html).
