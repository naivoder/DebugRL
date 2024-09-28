import gymnasium as gym
import numpy as np


class OneActionZeroObsEnv(gym.Env):
    def __init__(self):
        super(OneActionZeroObsEnv, self).__init__()
        self.action_space = gym.gym.spaces.Discrete(1)  # Only one action
        self.observation_space = gym.spaces.Box(
            low=0, high=0, shape=(1,), dtype=np.float32
        )
        self.state = np.array([0], dtype=np.float32)
        self.done = False

    def reset(self):
        self.done = False
        return self.state

    def step(self, action):
        reward = 1.0  # +1 reward every timestep
        self.done = True  # Episode ends after one timestep
        info = {}
        return self.state, reward, self.done, info

    def render(self, mode="human"):
        pass

    def close(self):
        pass


class OneActionRandomObsEnv(gym.Env):
    def __init__(self):
        super(OneActionRandomObsEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(1)  # Only one action
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(1,), dtype=np.int32
        )
        self.done = False

    def reset(self):
        self.done = False
        self.state = np.array([np.random.choice([-1, 1])], dtype=np.int32)
        return self.state

    def step(self, action):
        reward = float(self.state[0])  # Reward matches the observation
        self.done = True  # Episode ends after one timestep
        info = {}
        return self.state, reward, self.done, info

    def render(self, mode="human"):
        pass

    def close(self):
        pass


class OneActionTwoObsEnv(gym.Env):
    def __init__(self):
        super(OneActionTwoObsEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(1)  # Only one action
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )
        self.current_step = 0
        self.done = False

    def reset(self):
        self.current_step = 0
        self.done = False
        self.state = np.array([0], dtype=np.float32)  # Start with observation 0
        return self.state

    def step(self, action):
        if self.current_step == 0:
            reward = 0.0
            self.state = np.array([1], dtype=np.float32)  # Next observation is 1
            self.done = False
            self.current_step += 1
        else:
            reward = 1.0  # +1 reward at the end
            self.done = True
            self.current_step += 1
        info = {}
        return self.state, reward, self.done, info

    def render(self, mode="human"):
        pass

    def close(self):
        pass


class TwoActionsZeroObsEnv(gym.Env):
    def __init__(self):
        super(TwoActionsZeroObsEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(2)  # Two actions
        self.observation_space = gym.spaces.Box(
            low=0, high=0, shape=(1,), dtype=np.float32
        )
        self.state = np.array([0], dtype=np.float32)
        self.done = False

    def reset(self):
        self.done = False
        return self.state

    def step(self, action):
        if action == 0:
            reward = 1.0  # +1 reward for action 0
        elif action == 1:
            reward = -1.0  # -1 reward for action 1
        else:
            raise ValueError("Invalid action")
        self.done = True  # Episode ends after one timestep
        info = {}
        return self.state, reward, self.done, info

    def render(self, mode="human"):
        pass

    def close(self):
        pass


class TwoActionsRandomObsEnv(gym.Env):
    def __init__(self):
        super(TwoActionsRandomObsEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(2)  # Two actions
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(1,), dtype=np.int32
        )
        self.done = False

    def reset(self):
        self.done = False
        self.state = np.array([np.random.choice([-1, 1])], dtype=np.int32)
        return self.state

    def step(self, action):
        obs = self.state[0]
        # Map action 0 to -1 and action 1 to +1
        action_value = -1 if action == 0 else 1
        if action_value == obs:
            reward = 1.0  # +1 reward if action matches observation
        else:
            reward = -1.0  # -1 reward otherwise
        self.done = True  # Episode ends after one timestep
        info = {}
        return self.state, reward, self.done, info

    def render(self, mode="human"):
        pass

    def close(self):
        pass
